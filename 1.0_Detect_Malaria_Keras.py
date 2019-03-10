#Import some useful libraries
import numpy as np #array handling and linear algebra
import matplotlib.pyplot as plt #plotting lib

#Data loading
from PIL import Image
import cv2 as cv
import os

#Machine learning framework
# =============================================================================
# import keras
# from sklearn.model_selection import train_test_split
# from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, AveragePooling2D, Dropout, Input
# from keras.layers.normalization import BatchNormalization
# from keras.layers.merge import concatenate
# from keras import Model, Sequential
# 
# =============================================================================
import keras
from keras.models import Sequential 

# Convolution is sequential
# Since we are going to work on images which are 2 dimentional unlike videos (time)
# API update use Conv2D instead of Convolution2D
from keras.layers import Conv2D 

# Step 2 Pooling step
from keras.layers import MaxPooling2D

# Avoid overfitting, import Dropout
from keras.layers import Dropout

# Step 3 Flatten
from keras.layers import Flatten

# Add fully connected layers in an ANN
from keras.layers import Dense

print("Importing finished!")


''' We are inside cell_images folder'''
# Load Data
print(os.getcwd())

print(os.listdir('..\\cell_images'))
# print(os.listdir("../cell_images")) for linux

parasitized_path = os.listdir("..\\cell_images\\Parasitized")
print(len(parasitized_path)) # show all paths = 13780
print(parasitized_path[1]) # show 1th path
#print(parasitized_path[:10]) # show first 10 paths

# Rename
#infected_path = parasitized_path

uninfected_path = os.listdir("..\\cell_images\\Uninfected")
print(len(uninfected_path)) # 13780
print(uninfected_path[1])


# read one image using openCV
#image_single = cv.imread('..\\cell_images\\Parasitized\\'+ parasitized_path[1])
#cv.imshow("image_single", image_single)
#cv.waitKey(0)
#cv.destroyAllWindows()


data = []
labels = []

# Read all images inside Parasitized Path
for uninfected in uninfected_path:
    try:
        image = cv.imread('..\\cell_images\\Uninfected\\' + uninfected)        
        
        # Convert to PIL array 
        img_pil_array = Image.fromarray(image, 'RGB')
        
        # Resize image
        img_resized_np = img_pil_array.resize((64, 64))
        
        # append to data
        data.append(np.array(img_resized_np))
        
        # Label the image as 1 = uninfected
        labels.append(0)
        
    except AttributeError:
        print('Error exception uninfected_path')

for parasitized in parasitized_path:
    try:
        image = cv.imread('..\\cell_images\\Parasitized\\'+ parasitized)
        # Convert to numpy array
        img_pil_array = Image.fromarray(image, 'RGB')
        
        # Resize all images to get same size
        img_resized_np = img_pil_array.resize((64, 64))
        
        # alternative approach: using sklearn to resize   
        #from skimage.transform import resize
        #img_resized_sklearn = resize(image, (64, 64), anti_aliasing=True)
    
        # or you can resize image using openCV
        # you need to convert it to an array then, you can append to data array
        #img_resized_opncv = cv.resize(image, dsize=(64, 64), interpolation=cv.INTER_CUBIC)
        
        # append all images into single array    
        data.append(np.array(img_resized_np))
        
        '''
        How can we track parasitized and normal?
        We are using all parasitized as label 1
        and all uninfected as label 1
        So, if the label is 1, it is parasitized
        '''
        labels.append(1) 
    except AttributeError:
        print('Error exeption parasitized_path')
    
#print(data[:2])
'''
To do    
1. Use openCV to resize image
2. Check PIL vs OpenCV for resize and array conversion

'''
print(data[1]) # numpy array
print(labels[1]) # 0

print(len(data)) # 27558
print(len(labels)) # 27558


#Shape of the data
# data = np.array(data)
# labels = np.array(labels)

print("Shape of the data array: ", np.shape(data))
print("Shape of the label array: ", np.shape(labels))


# Save image array to use later. Made it easy
cells = np.array(data)
labels = np.array(labels)

np.save('Cells' , cells)
np.save('Labels' , labels)


print('Cells : {} | labels : {}'.format(cells.shape , labels.shape))

print(cells.shape) # (27558, 64, 64, 3)
print(cells.shape[0]) # 27558

plt.figure(1 , figsize = (15 , 9))
n = 0 
for i in range(49):
    n += 1 
    
    # Take random image
    r = np.random.randint(0 , cells.shape[0] , 1)
    
    plt.subplot(7 , 7 , n)
    plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)
    
    plt.imshow(cells[r[0]])
    
    plt.title('{} : {}'
              .format('Infected' if labels[r[0]] == 1 
                                  else 'Unifected' , labels[r[0]]) )
    plt.xticks([]) , plt.yticks([])
    
plt.show()

plt.figure(1, figsize = (15, 9))
n = 0
for i in range(16):
    
    n += 1
    
    # Take a random number in each iteration
    # np.random.randint(low, high, dtType)
    r = np.random.randint(0, cells.shape[0], 1)
    
    # Create subplot
    plt.subplot(4, 4, n)
    
    # Adjust subplots
    plt.subplots_adjust(hspace = 0.2, wspace = 0.2)
    
    # Show single image using random selection
    # For each iteration, random number will be selected
    # image will be shown according to random numbered
    plt.imshow(cells[ r[0] ])
    
    
    # Show title
    plt.title('{} : {}'.format(
            'Infected' if labels[r[0]] == 1
            else 'Uninfected', labels[r[0]]
            ))
    
    plt.xticks([]), plt.yticks([])
plt.show()


print(cells.shape) # 27558, 64, 64, 3
plt.figure(1, figsize = (10 , 7))
plt.subplot(1 , 2 , 1)
plt.imshow(cells[0])
plt.title('{} : {}'.format(
            'Infected' if labels[0] == 1
            else 'Uninfected', labels[0]
            ))
plt.xticks([]) , plt.yticks([])

plt.subplot(1 , 2 , 2)
plt.imshow(cells[26356])
plt.title('{} : {}'.format(
            'Infected' if labels[26356] == 1
            else 'Uninfected', labels[26356]
            ))
plt.xticks([]) , plt.yticks([])
plt.show()



# Load from the saved numpy array cells and labels
cells_loaded=np.load("Cells.npy")
labels_loaded=np.load("Labels.npy")

print(cells_loaded.shape[0]) # 27558

# Arrange 
shuffled_cells = np.arange(cells_loaded.shape[0])
print(shuffled_cells)

# Random shuffle
np.random.shuffle(shuffled_cells)
print(shuffled_cells)


print(cells_loaded[shuffled_cells])
print(labels_loaded[shuffled_cells])

cells_randomly_shuffled = cells_loaded[shuffled_cells]
labels_randomly_shuffled = labels_loaded[shuffled_cells]

print(np.unique(labels_randomly_shuffled)) # [0 1]
print(len(cells_randomly_shuffled)) # 27558

num_classes = len(np.unique(labels_randomly_shuffled))
print(num_classes)

len_data = len(cells_randomly_shuffled)
print(len_data)

print(0.1*len_data) # 2755.8

print(len(cells_randomly_shuffled[(int)(0.1*len_data):])) # 24803
print(len(cells_randomly_shuffled[:(int)(0.1*len_data)])) # 2755


''' Train Test Split Technique 1'''
(x_train, x_test) = cells_randomly_shuffled[(int)(0.1*len_data):], cells_randomly_shuffled[:(int)(0.1*len_data)]
print(len(x_train))
print(len(x_test))

# As we are working on image data we are normalizing data by divinding 255.
x_train = x_train.astype('float32')/255 
print(x_train)

x_test = x_test.astype('float32')/255

x_train_len = len(x_train)
x_test_len = len(x_test)
print(x_train_len)
print(x_test_len)

(y_train, y_test) = labels_randomly_shuffled[(int)(0.1*len_data):], labels_randomly_shuffled[:(int)(0.1*len_data)]


#Doing One hot encoding as classifier has multiple classes
y_train=keras.utils.to_categorical(y_train,num_classes)
y_test=keras.utils.to_categorical(y_test,num_classes)

print(y_train)
print(len(y_train))
print(y_test)
print(len(y_test))



# Create model
model = Sequential()

'''
api reference r1.13
tf.layers.conv2d(
    inputs,
    filters,
    kernel_size,
    strides=(1, 1),
    padding='valid',
    data_format='channels_last',
    dilation_rate=(1, 1),
    activation=None,
    use_bias=True,
    kernel_initializer=None,
    bias_initializer=tf.zeros_initializer(),
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    trainable=True,
    name=None,
    reuse=None
    
    
)

What does conv2D do in tensorflow?
https://stackoverflow.com/questions/34619177/what-does-tf-nn-conv2d-do-in-tensorflow

'''
# Add hidden layer
model.add(Conv2D(filters=16, kernel_size=2, padding='same', activation='relu', input_shape=(64,64,3)))

# Step 2 - Max Pooling - Taking the maximum
# Why? Reduce the number of nodes for next Flattening step
model.add(MaxPooling2D(pool_size = (2, 2)))

# Add another hidden layer
model.add(Conv2D(filters=16, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))


model.add(Conv2D(filters=16, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))

# Droput to avoid overfitting
model.add(Dropout(0.2))

# Step 3 - Flatten - huge single 1 dimensional vector
model.add(Flatten())

# Step 4 - Full Connection
# output_dim/units: don't take too small, don't take too big
# common practice take a power of 2, such as 128, 256, etc.
model.add(Dense(units = 128, activation = "relu"))
model.add(Dense(units = 2, activation = "softmax")) 

model.summary()


# Step 5
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# Step 6 Fitting model
model.fit(x_train, y_train, batch_size=50, epochs=20, verbose=1)

# Check accuracy
accuracy = model.evaluate(x_test, y_test, verbose=1)

# Save model weights
from keras.models import load_model
model.save('malaria_tfcnnsoftmax_category.h5')

# Use model using tkinter
from keras.models import load_model
from PIL import Image
import numpy as np
import os
import cv2 as cv

def convert_to_array(img):
    img = cv.imread(img)
    img = Image.fromarray(img)
    img = img.resize(64, 64)
    return np.array(img)

def get_label(label):
    if label == 0:
        return 'Uninfected'
    if label == 1:
        return 'Parasitized'

def predict_malaria(img_file):
    
    model = load_model('malaria_tfcnnsoftmax_category.h5')
    
    print('Predciting Malaria....')
    
    img_array = convert_to_array(img_file)
    img_array = img_array/255
    
    img_data = []
    img_data.append(img_array)
    img_data = np.array(img_data)
    
    score = model.predict(img_data, verbose=1)
    print('Score', score)
    
    label_index = np.argmax(score)
    
    result = get_label(label_index)
    return result, 'Predicted image is : ' + result + 'with accuracy = ' + str(accuracy)


"""from tkinter import Frame, Tk, BOTH, Text, Menu, END
from tkinter import filedialog 
from tkinter import messagebox as mbox

class Example(Frame):

    def __init__(self):
        super().__init__()   

        self.initUI()


    def initUI(self):

        self.master.title("File dialog")
        self.pack(fill=BOTH, expand=1)

        menubar = Menu(self.master)
        self.master.config(menu=menubar)

        fileMenu = Menu(menubar)
        fileMenu.add_command(label="Open", command=self.onOpen)
        menubar.add_cascade(label="File", menu=fileMenu)        

        

    def onOpen(self):

        ftypes = [('Image', '*.png'), ('All files', '*')]
        dlg = filedialog.Open(self, filetypes = ftypes)
        fl = dlg.show()
        c,s=predict_cell(fl)
        root = Tk()
        T = Text(root, height=4, width=70)
        T.pack()
        T.insert(END, s)
        

def main():

    root = Tk()
    ex = Example()
    root.geometry("100x50+100+100")
    root.mainloop()  


if __name__ == '__main__':
    main()"""