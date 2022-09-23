import tkinter as tk
from PIL import Image, ImageTk
import tkinter.filedialog as fd
from runTracking import runTracking
import tkinter.font as font
from tkinter import ttk

########################################################################################################################
########################################################################################################################
########################################################################################################################
class ToolTip(object):
    def __init__(self,widget):
        self.widget = widget
        self.tipwindow = None
        self.id = None
        self.x=self.y=0

    def showtip(self,text):
        "Display text in tooltip window"
        self.text=text
        if self.tipwindow or not self.text:
            return
        x,y,cx,cy=self.widget.bbox("insert")
        y = y + cy + self.widget.winfo_rooty() + 27
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(1)
        tw.wm_geometry("+%d+%d" % (x, y))
        label = tk.Label(tw, text=self.text, justify='left',
                      background="#ffffe0", relief='solid', borderwidth=1,
                      font=("tahoma", "12", "normal"))
        label.pack(ipadx=1)

    def hidetip(self):
        tw=self.tipwindow
        self.tipwindow=None
        if tw:
            tw.destroy()
def CreateToolTip(widget, text):
    toolTip = ToolTip(widget)
    def enter(event):
        toolTip.showtip(text)
    def leave(event):
        toolTip.hidetip()
    widget.bind('<Enter>', enter)
    widget.bind('<Leave>', leave)


########################################################################################################################
########################################################################################################################
########################################################################################################################

window = tk.Tk()
window.rowconfigure(0, weight=1)
window.columnconfigure(0, weight=1)
window.wm_state('normal')

homepage = tk.Frame(window)
seg_page = tk.Frame(window)
track_page = tk.Frame(window)
fam_page = tk.Frame(window)
analysis_page = tk.Frame(window)
help_page = tk.Frame(window)

train_page = tk.Frame(window)
test_page = tk.Frame(window)
predict_page = tk.Frame(window)

def_font = font.Font(family='Times')
for frame in (homepage, seg_page, track_page, fam_page, help_page, train_page, test_page, predict_page):
    frame.grid(row=0, column=0, sticky='nsew')


def show_frame(frame):
    frame.tkraise()


show_frame(homepage)

window.title('MOSTHyFI - Segmentation, Tracking and Analysis')

window.geometry("1000x900")

my_menu = tk.Menu(window)
window.config(menu=my_menu)
my_menu.config(font=(def_font, 15))

file_menu = tk.Menu(my_menu)
ana_menu = tk.Menu(my_menu)

seg_menu = tk.Menu(my_menu)

my_menu.add_cascade(label='File', menu=file_menu)
my_menu.add_separator()
my_menu.add_command(label='Home', command=lambda : show_frame(homepage))
my_menu.add_separator()
my_menu.add_cascade(label='Segmentation', menu=seg_menu)  # , command=lambda: show_frame(seg_page))
#
seg_menu.add_command(label='Segmentation Home', command=lambda: show_frame(seg_page), font=('Arial', 15))
seg_menu.add_separator()
seg_menu.add_command(label='Train', command=lambda: show_frame(train_page), font=('Arial', 15))
seg_menu.add_command(label='Test', command=lambda: show_frame(test_page), font=('Arial', 15))
seg_menu.add_command(label='Predict', command=lambda: show_frame(predict_page), font=('Arial', 15))
#
my_menu.add_separator()
my_menu.add_command(label='Tracking', command=lambda: show_frame(track_page))
my_menu.add_separator()
my_menu.add_command(label='Family Tree', command=lambda: show_frame(fam_page))
my_menu.add_separator()
my_menu.add_cascade(label='Analysis', menu=ana_menu)
my_menu.add_separator()
my_menu.add_command(label='Help', command=lambda: show_frame(help_page))

file_menu.add_command(label='New', font=('Arial', 15))
file_menu.add_command(label='Open', font=('Arial', 15))
file_menu.add_command(label='Save', font=('Arial', 15))
file_menu.add_command(label='Save as', font=('Arial', 15))

# seg_menu.add_command(label='Segmentation',command=lambda: show_frame(seg_page))
# app_menu.add_command(label='Quit!', command=window.quit, background='red')
# app_menu.add_command(label='Home', command=lambda: show_frame(homepage), background='green', font=('Arial', 15))

ana_menu.add_command(label='Cross-Correlation', font=('Arial', 15))
ana_menu.add_command(label='Other Analysis', font=('Arial', 15))
ana_menu.add_command(label='Analysis III', font=('Arial', 15))

# ----------------------------------------------------------------------------------------------------------------------
# Homepage
homepage_greet = tk.Label(homepage, text='WELCOME TO FIAPP', font=('Times', 40, 'bold'))
homepage_greet.place(x=50, y=100)
buttonH1 = tk.Button(homepage, text="Segmentation", width=15, command=lambda: show_frame(seg_page), font=('Times', 20))

buttonH2 = tk.Button(homepage, text="Tracking",  width=15,
                    command=lambda: show_frame(track_page), font=('times', 20))
buttonH3 = tk.Button(homepage,  width=15, text="Family Tree", command=lambda: show_frame(fam_page), font=('times', 20))
buttonH4 = tk.Button(homepage,  width=15, text="Analysis", command=lambda: show_frame(analysis_page), font=('times', 20))
buttonH1.place(x=200, y=200)
buttonH2.place(x=200, y=300)
buttonH3.place(x=200, y=400)
buttonH4.place(x=200, y=500)


CreateToolTip(buttonH2, text = 'Click here for tracking!')
# ----------------------------------------------------------------------------------------------------------------------
# Segmentation page

seg_page_greet = tk.Label(seg_page, text='Segmentation', font=('Arial', 40))
seg_page_greet.place(x=50, y=100)

# label1 = tk.Label(seg_page, text='Model: ', font=('times', 20))
# label1.place(x=100, y=200)
# modelNameList = ['FC-DenseNet', 'MobileUNet3D-Skip', 'ResNet-101', 'Encoder_Decoder3D', 'Encoder_Decoder3D_contrib',
#                  'DeepLabV33D', 'FRRN-A', 'FRRN-B', 'FCN8', 'GCN-Res50', 'GCN-Res101', 'GCN-Res152', 'AdapNet3D',
#                  'ICNet-Res50', 'ICNet-Res101', 'ICNet-Res152', 'PSPNet-Res50', 'PSPNet-Res101', 'PSPNet-Res152']
# modelName = tk.StringVar()
# modelName.set('FC-DenseNet')
# modelMenu = tk.OptionMenu(seg_page, modelName, *modelNameList)
# modelMenu.config(font=('helvetica', 20, 'bold'))
# modelMenu.place(x=220, y=200)

button1 = tk.Button(seg_page, text="<- Back", command=lambda: show_frame(homepage), font=('times', 20))
button2 = tk.Button(seg_page, width=12, text="Train", command=lambda: show_frame(train_page), font=('times', 20))
button3 = tk.Button(seg_page, width=12, text="Test", command=lambda: show_frame(track_page), font=('times', 20))
button4 = tk.Button(seg_page, width=12, text="Predict", command=lambda: show_frame(predict_page), font=('times', 20))

button1.place(x=50, y=50)
button2.place(x=100, y=300)
button3.place(x=100, y=400)
button4.place(x=100, y=500)

############
gt_path = ''
train_output_path = ''
predictOutputPath = tk.StringVar()
epochs = 0
imageName = tk.StringVar()
trainModelName = tk.StringVar()
gtp = tk.StringVar()
tol = tk.StringVar()


def browseGT():
    try:
        gtp.set(fd.askdirectory())
        gt_path = gtp
    except:
        print('star!')
        gt_path = '/default/gt/path/'


def browseImage():
    imageName.set(fd.askopenfilename(defaultextension=".nii",filetypes=[("NIFTI Files","*.nii")]))
    # gtp.set(imageName)


def trainOutputLocation():
    try:
        tol.set(fd.askdirectory())
        train_output_path = tol
    except:
        train_output_path = '/default/train/output/path/'


def predictOutputLocation():
    try:
        predictOutputPath.set(fd.askdirectory())
    except:
        predict_output_path = '/default/predict/output/path/'


def callTrain(modl, epochs, grtp, opp):
    trainModelName.set(modl)
    # segment_train(model=modl, epochs=epochs, gt_path=grtp, output_path=opp)


def display_selected(choice):
    choice = modelName.get()
    print(choice)
##########################
train_page_greet = tk.Label(train_page, text='Train', font=('Arial', 40, 'bold'))
train_page_greet.place(x=50, y=100)

numEpochs = tk.IntVar()
label2 = tk.Label(train_page, text="No. of Epochs", font=('times', 20, 'bold'))
entry1 = tk.Entry(train_page, textvariable=numEpochs, font=('times', 20, 'bold'), width=14)

button1 = tk.Button(train_page, text="<- Back", command=lambda: show_frame(seg_page), font=('times', 20, 'bold'))

button2 = tk.Button(train_page,  width=26, text="Select Training Data Folder", font=('times', 20, 'bold'), command=lambda: browseGT())
entry2 = ttk.Entry(train_page, textvariable=gtp, width=30, font=('times',20))

button3 = tk.Button(train_page, width=20, text="Select Output Folder", command=lambda: trainOutputLocation(),
                    font=('times', 20, 'bold'))
entry3 = ttk.Entry(train_page, textvariable=tol, width=30, font=('times',20))
#
modelNameList = ['FC-DenseNet', 'MobileUNet3D-Skip', 'ResNet-101', 'Encoder_Decoder3D', 'Encoder_Decoder3D_contrib',
                 'DeepLabV33D', 'FRRN-A', 'FRRN-B', 'FCN8', 'GCN-Res50', 'GCN-Res101', 'GCN-Res152', 'AdapNet3D',
                 'ICNet-Res50', 'ICNet-Res101', 'ICNet-Res152', 'PSPNet-Res50', 'PSPNet-Res101', 'PSPNet-Res152']
modelName = tk.StringVar()
modelName.set(modelNameList[0])
modelMenu = tk.OptionMenu(train_page, modelName, *modelNameList, command=display_selected)
modelMenu.config(font=('helvetica', 20, 'bold'))
#
button4 = tk.Button(train_page, width=10, text="check",
                    command=lambda: print(modelName.get() + '\n' + str(numEpochs.get()) + '\n' + gtp.get() + '\n' + tol.get()),
                    font=('times', 20, 'bold'))
button5 = tk.Button(train_page, width=10, text="RUN", background="red", foreground="white",
                    command=lambda: callTrain(modelName.get(), numEpochs.get(), gtp.get(), tol.get()), font=('times', 20, 'bold'))
## dropdown
label1 = tk.Label(train_page, text='Model: ', font=('times', 20, 'bold'))
label1.place(x=100, y=200)

modelMenu.place(x=500, y=200)
##
button1.place(x=50, y=50)
label2.place(x=100, y=300)
entry1.place(x=500, y=300)
button2.place(x=100, y=400)
button3.place(x=100, y=500)
button4.place(x=100, y=600)
button5.place(x=100, y=700)

entry2.place(x=500, y=400)
entry3.place(x=500, y=500)



# button6 = tk.Button(train_page, text='printModelName', command=lambda: print(modelName.get()))
# button6.place(x=400, y= 800)
###################################################

startTime = 1
endTime = 41
predModelName = tk.StringVar()


def callPredict(model, st, et):
    predModelName.set(model)
    startTime = st
    endTime = et
    # segment_predict(predModelName, imageName, predict_output_path, startTime, endTime)


predict_page_greet = tk.Label(predict_page, text='Predict', font=('Arial', 40, 'bold'))
predict_page_greet.place(x=50, y=100)

## dropdown
label1 = tk.Label(predict_page, text='Model: ', font=('times', 20, 'bold'))
label1.place(x=100, y=200)

modelNameList = ['FC-DenseNet', 'MobileUNet3D-Skip', 'ResNet-101', 'Encoder_Decoder3D', 'Encoder_Decoder3D_contrib',
                 'DeepLabV33D', 'FRRN-A', 'FRRN-B', 'FCN8', 'GCN-Res50', 'GCN-Res101', 'GCN-Res152', 'AdapNet3D',
                 'ICNet-Res50', 'ICNet-Res101', 'ICNet-Res152', 'PSPNet-Res50', 'PSPNet-Res101', 'PSPNet-Res152']
modelName = tk.StringVar()
modelName.set('FC-DenseNet')
modelMenu = tk.OptionMenu(predict_page, modelName, *modelNameList)
modelMenu.config(font=('helvetica', 20, 'bold'))
modelMenu.place(x=500, y=200)
##

button1 = tk.Button(predict_page, text="<- Back", command=lambda: show_frame(seg_page), font=('times', 20, 'bold'))

button2 = tk.Button(predict_page, text="Select Image", command=lambda: browseImage(), font=('times', 20, 'bold'))
entryPr1 = tk.Entry(predict_page, textvariable=imageName, font=('times', 20, 'bold'))
button3 = tk.Button(predict_page, text="Select Output Folder", command=lambda: predictOutputLocation(),
                    font=('times', 20, 'bold'))
entryPr2 = tk.Entry(predict_page, textvariable=predictOutputPath, font=('times', 20, 'bold'))
startT = tk.IntVar()
endT = tk.IntVar()
label2 = tk.Label(predict_page, text="Start Time", font=('times', 20, 'bold'))
entry1 = tk.Entry(predict_page, textvariable=startT, font=('times', 20, 'bold'))
label3 = tk.Label(predict_page, text='End Time', font=('times', 20, 'bold'))
entry2 = tk.Entry(predict_page, textvariable=endT, font=('times', 20, 'bold'))
button4 = tk.Button(predict_page, text="check",
                    command=lambda: print(modelName.get(), imageName, predictOutputPath, startT.get(), endT.get()),
                    font=('times', 20, 'bold'))
button5 = tk.Button(predict_page, text="RUN", background="red", foreground="white",
                    command=lambda: callPredict(modelName.get(), startT.get(), endT.get()), font=('times', 20, 'bold'))

button1.place(x=50, y=50)
label2.place(x=100, y=300)
entry1.place(x=500, y=300)
label3.place(x=100, y=400)
entry2.place(x=500, y=400)
button2.place(x=100, y=500)
button3.place(x=100, y=600)
button4.place(x=100, y=700)
button5.place(x=100, y=800)

entryPr1.place(x=500, y=500)
entryPr2.place(x=500, y=600)

################ TRACKING ###########################################################################################

imgname = tk.StringVar()
segloc = tk.StringVar()
trackloc = tk.StringVar()

strT = tk.IntVar()
enT = tk.IntVar()
trbT = tk.IntVar()
ost = tk.IntVar()
p1n = tk.StringVar()
p2n = tk.StringVar()

def segOPfolder():
    segloc.set(fd.askdirectory())
def trackOPfolder():
    trackloc.set(fd.askdirectory())
def browseImageTr():
    imgname.set(fd.askopenfilename(defaultextension='.nii', filetypes=[("NIFTI Files","*.nii")]))

# def callTracking():
#     tracking()


track_page_greet = tk.Label(track_page, text='Tracking', font=('Arial', 40, 'bold'))
track_page_greet.place(x=50, y=100)


buttonTr1 = tk.Button(track_page, text="<- Back", command=lambda: show_frame(homepage), font=('times', 15, 'bold'))

buttonTr2 = tk.Button(track_page, text="Select Image", command=lambda: browseImageTr(), font=('times', 15, 'bold'))
buttonTr3 = tk.Button(track_page, text="Select Seg. Output Folder", command=lambda: segOPfolder(),font=('times', 15, 'bold'))
buttonTr4 = tk.Button(track_page, text ="Select Track Output Folder", command=lambda: trackOPfolder(), font=('Times',15,'bold'))
buttonTr5 = tk.Button(track_page, text="Check", command=lambda: print("Check check check"))
buttonTr6 = tk.Button(track_page, text="Run", command=lambda: runTracking(imageName=imgname.get(),
                    segmentationOPFolder=segloc.get(), trackingOPFolder=trackloc.get(), startTime=strT.get(),
                    endTime=enT.get(), trackbackTime=trbT.get(),min_obj_size=ost.get(), protein1Name=p1n.get(),
                    protein2Name=p2n.get()), background='red')

labelTr1 = tk.Label(track_page, text="Start Time", font=('Times',15,'bold'))
labelTr2 = tk.Label(track_page, text="End Time", font=('Times',15,'bold'))
labelTr3 = tk.Label(track_page, text="Min size threshold", font=('Times', 15, 'bold'))
labelTr4 = tk.Label(track_page, text="Trackback Time", font=('Times',15,'bold'))
# labelTr5 = tk.Label(track_page, text="Protein 1 Name", font=('Times',15,'bold'))
# labelTr6 = tk.Label(track_page, text="Protein 2 Name", font=('Times',15,'bold'))

entryTr1 = ttk.Entry(track_page, textvariable=imgname, font=('Times',15,'bold'))
entryTr2 = ttk.Entry(track_page, textvariable=segloc, font=('Times',15,'bold'))
entryTr3 = ttk.Entry(track_page, textvariable=trackloc, font=('Times',15,'bold'))

entryTr4 = ttk.Entry(track_page, textvariable=strT, font=('Times',15,'bold'))
entryTr4.insert(1,'1')
entryTr5 = ttk.Entry(track_page, textvariable=enT, font=('Times',15,'bold'))
entryTr5.insert(1,'41')
entryTr6 = ttk.Entry(track_page, textvariable=trbT, font=('Times',15,'bold'))
entryTr6.insert(1,'2')
entryTr7 = ttk.Entry(track_page, textvariable=ost, font=('Times',15,'bold'))
entryTr7.insert(1,'27')
# entryTr8 = ttk.Entry(track_page, textvariable=p1n, font=('Times',15,'bold'))
# entryTr9 = ttk.Entry(track_page, textvariable=p2n, font=('Times',15,'bold'))

buttonTr1.place(x=50,y=50)
buttonTr2.place(x=100,y=200); entryTr1.place(x=400,y=200, width=400)
buttonTr3.place(x=100,y=250); entryTr2.place(x=400,y=250, width=400)
buttonTr4.place(x=100,y=300); entryTr3.place(x=400,y=300, width=400)

labelTr1.place(x=100, y=350); entryTr4.place(x=400, y=350, width=100)
labelTr2.place(x=100, y=400); entryTr5.place(x=400, y=400, width=100)
labelTr3.place(x=100, y=450); entryTr7.place(x=400, y=450, width=100)
labelTr4.place(x=100, y=500); entryTr6.place(x=400, y=500, width=100)
# labelTr5.place(x=100, y=550); entryTr8.place(x=400, y=550, width=100)
# labelTr6.place(x=100, y=600); entryTr9.place(x=400, y=600, width=100)

buttonTr5.place(x=100, y=700)
buttonTr6.place(x=100, y=750)

CreateToolTip(track_page_greet, text='This page is used for tracking the objects \n '
                                     'based on the original image and the segmentation output.\n'
                                     'You need to select the following:\n'
                                     '\t The image of interest\n'
                                     '\t The folder that contains the output of segmentation process\n'
                                     '\t A folder to save the output of tracking\n'
                                     '\t Time frame to start traking (default 1)\n'
                                     '\t Time frame to stop tracking (default 41)\n'
                                     '\t Minimum size of the objects to consider for tracking (default 27)\n'
                                     '\t Trackback Time (How many time frames do you want to look for the objects to track (default 2)'
                                     )


########################################################################################################################

test_page_greet = tk.Label(test_page, text='Test', font=('Arial', 40, 'bold'))
test_page_greet.place(x=50, y=100)

fam_page_greet = tk.Label(fam_page, text='Family Tree and LPR', font=('Arial', 40, 'bold'))
fam_page_greet.place(x=50, y=100)

help_page_greet = tk.Label(help_page, text='Help', font=('Arial', 40, 'bold'))
help_page_greet.place(x=50, y=100)

# ----------------------------------------------------------------------------------------------------------------------


window.mainloop()
