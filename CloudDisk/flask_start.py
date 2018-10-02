# -*- coding: utf-8 -*-
import os

from flask import Flask, render_template, url_for
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField

app = Flask(__name__)
app.config['SECRET_KEY'] = 'I have a dream'
app.config['UPLOADED_PHOTOS_DEST'] = os.getcwd()

photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)
patch_request_class(app)  # set maximum file size, default is 16MB


class UploadForm(FlaskForm):
    photo = FileField(validators=[
        FileAllowed(photos, u'只能上传图片！'),
        FileRequired(u'文件未选择！')])
    submit = SubmitField(u'上传')


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    form = UploadForm()
    if form.validate_on_submit():
        filename = photos.save(form.photo.data)
        file_url = photos.url(filename)
    else:
        file_url = None
    return render_template('index.html', form=form, file_url=file_url)

@app.route('/python', methods=['GET', 'POST'])
def python_file():
    return render_template('python.html')

@app.route('/python_arith', methods=['GET', 'POST'])
def pythonarith_file():
    return render_template('python_arith.html')

@app.route('/python_list', methods=['GET', 'POST'])
def pythonlist_file():
    return render_template('python_list.html')

if __name__ == '__main__':
    app.run(host = '0.0.0.0')