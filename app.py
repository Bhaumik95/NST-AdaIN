import os
import torch
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename
from wtforms import FileField, SubmitField, FloatField, HiddenField
from wtforms.validators import InputRequired
from PIL import Image
from torchvision import transforms
import io

# import your existing AdaIN code
from utils.models import VGGEncoder, Decoder
from utils.utils import adaptive_instant_normalization, calc_mean_std

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secretkey'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
Bootstrap(app)

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class UploadForm(FlaskForm):
    content = FileField('Content Image')
    style = FileField('Style Image')
    content_path = HiddenField()
    style_path = HiddenField()
    alpha = FloatField('Alpha', default=1.0)
    submit = SubmitField('Transfer Style')
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    

encoder = VGGEncoder('vgg_normalised.pth').to(device)
decoder = Decoder().to(device)

# decoder.load_state_dict(torch.load('experiment/train_1/checkpoint_epoch_135.pth',map_location=device))
checkpoint = torch.load(
    'experiment/train_1/checkpoint_epoch_135.pth',
    map_location=device
)

decoder.load_state_dict(checkpoint['decoder_state_dict'])


encoder.eval() 
decoder.eval()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def style_transfer(content_image,style_image,encoder,decoder,alpha,device):
    content_transform = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor()
    ])

    style_transform = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor()
    ])
    
    content_image = content_transform(content_image).unsqueeze(0).to(device)
    style_image = style_transform(style_image).unsqueeze(0).to(device)
    
    with torch.no_grad():    
        content_feat = encoder(content_image, is_test=True)
        style_feat = encoder(style_image, is_test=True)
        
        stylized_feat = adaptive_instant_normalization(content_feat,style_feat)
        
        stylized_feat = alpha * stylized_feat + (1-alpha) * content_feat
        
        stylized_image = decoder(stylized_feat)
    
    return stylized_image


def save_image(image_tensor,path):
    image_tensor = image_tensor.cpu().clone()
    image_tensor = image_tensor.squeeze(0)
    image_tensor = image_tensor.clamp(0,1)
    image = transforms.ToPILImage()(image_tensor)
    image.save(path)

@app.route('/', methods=['GET','POST'])
def index():
    form = UploadForm()
    result_image = None
    content_filename = None
    style_filename = None
    error = None
     
    if form.validate_on_submit():
        content_file = form.content.data
        style_file = form.style.data
        
        if content_file and allowed_file(content_file.filename):
            content_filename = secure_filename(content_file.filename)
            content_path = os.path.join(app.config['UPLOAD_FOLDER'],content_filename)
            content_file.save(content_path)
            form.content_path.data = content_path
        else:
            # content_filename = form.content_path.data 
            content_path = form.content_path.data
            if content_path:
                content_filename = os.path.basename(content_path)
                
        if style_file and allowed_file(style_file.filename):    
            style_filename = secure_filename(style_file.filename)
            style_path = os.path.join(app.config['UPLOAD_FOLDER'],style_filename)
            style_file.save(style_path)
            form.style_path.data = style_path
        else:
            # style_filename = form.style_path.data
            style_path = form.style_path.data
            if style_path:
                style_filename = os.path.basename(style_path)
                
        if content_filename and style_filename:
            try:
                content_image = Image.open(content_path).convert('RGB')
                style_image = Image.open(style_path).convert('RGB')
                alpha = float(form.alpha.data)
                
                stylized_image = style_transfer(content_image,style_image,encoder,decoder,alpha,device)
                
                result_filename = 'stylized_' + content_filename
                result_path = os.path.join(app.config['UPLOAD_FOLDER'],result_filename)
                
                save_image(stylized_image, result_path)
                
                result_image = result_filename
                
            except Exception as e:
                error = str(e)
                
    else:
        if not content_filename:
            error = 'Please upload content image'
        if not style_filename:
            error = 'Please upload style image'            
    
    return render_template('index.html', form=form, result_image = result_image, content_image = content_filename, 
                           style_image = style_filename, error=error)
    
@app.route('/uploads/<filename>')
def send_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/examples/<path:filename>')
def send_example(filename):
    return send_from_directory('examples', filename)

if __name__ == '__main__':
    from werkzeug.serving import run_simple
    run_simple('localhost', 5000, app, use_reloader=True, use_debugger=True)                 