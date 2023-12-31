import os
from flask import Flask, render_template, request
from flask_mysqldb import MySQL
from deeplearning import object_detection, video

app = Flask(__name__)
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'Vik@s123'  # Set your MySQL password here
app.config['MYSQL_DB'] = 'parking'

mysql = MySQL(app)

BASE_PATH = os.getcwd()
UPLOAD_PATH = os.path.join(BASE_PATH, 'static/uploads/')

@app.route('/', methods=['POST', 'GET'])
def index():
    cur = mysql.connection.cursor()

    if request.method == 'POST':
        upload_file = request.files['image_name']
        filename = upload_file.filename
        path_save = os.path.join(UPLOAD_PATH, filename)
        upload_file.save(path_save)

        if ".jpeg" in filename or ".png" in filename:
            text = object_detection(path_save, filename)
            name = ".jpeg" if ".jpeg" in filename else ".png"
        elif ".mp4" in filename:
            text = video(path_save, filename)
            name = ".mp4"

        getVals = list([val for val in text if val.isalpha() or val.isnumeric()])
        Number_detected = "".join(getVals)

        cur.execute("SELECT * FROM employee_data")
        fetchdata = cur.fetchall()
        cur.close()

        for i in fetchdata:
            a = (True if Number_detected in i else False)
            if a == True:
                User_Details = i
                user = 'Authorized'
                Eid = User_Details[1]
                Ename = User_Details[2]
                mob = User_Details[3]
                licence = User_Details[4]
                break
            else:
                User_Details = 'Unauthorized Person - Entry Denied'
                user = 'Unauthorized'
                Eid = None
                Ename = None
                mob = None
                licence = None

        return render_template('index.html', upload=True, upload_image=filename, text=Number_detected,
                               User_Details=User_Details, name=name, user=user, Eid=Eid, Ename=Ename, mob=mob,
                               licence=licence)

    return render_template('index.html', upload=False)

if __name__ == "__main__":
    app.run(debug=True)
