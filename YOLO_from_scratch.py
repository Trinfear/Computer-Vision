#!python3
# yolo implemented using opencv and numpy

import os
import cv2
import time
import argparse
import numpy as np

''' 

rewrite intake system:
    add in functionality to take in one image
        essentially same as what's currently here
        
    add in functionality to take in one dir of images
        if input is dir, break dir into batches of images
        feed each image in one at a time
        resave to a new directory
        
    add in functionality to take in videos
        break video into frames
        break frames into batches
        process each image
        reassamble into video
        save video

    add in funcitonality for real time videos?
        try to draw in real time
            intake each frame
            before rendering each frame pass to model
            draw on boxes
            pass frame to be drawn into the video

        try to have an idea of what's going on
            at a set time point, say once a second, take a frame
            run detection on the frame
            keep frame as current mapping until next frame


get cuda working
    make sure that the model uses gpu processing
    can this be done using cv2 or does it need to be transfered to keras or torch?

'''

## for using with command prompt
## no linebreaks below

## python YOLO_from_scratch.py
## -i="C:\Users\Tony\Desktop\Python Scripts\ML Practice\Computer V‭ision\YOLO Take Two\demo_img.png"
## -c="C:\Users\Tony\Desktop\Python Scripts\ML Practice\Computer V‭ision\YOLO Take Two\yolov3.cfg"
## -w="C:\Users\Tony\Desktop\Python Scripts\ML Practice\Computer V‭ision\YOLO Take Two\yolov3.weights"
## -cl="C:\Users\Tony\Desktop\Python Scripts\ML Practice\Computer V‭ision\YOLO Take Two\yolov3.txt"


def parse_cmd_line():
    # this sucks and needs fixing
    # lots of things here seem to cause errors? test out later
    ap = argparse.ArgumentParser()

    ap.add_argument('-i', '--image',
                    default=r"C:\Users\Tony\Desktop\Python Scripts\ML Practice\Computer V‭ision\YOLO Take Two\demo_img.png",
                    help = 'path to input image')

    ap.add_argument('-c', '--config',
                    default=r'C:\Users\Tony\Desktop\Python Scripts\ML Practice\Computer V‭ision\YOLO Take Two\yolov3.cfg',
                    help = 'path to yolo config file')

    ap.add_argument('-w', '--weights',
                    default=r'C:\Users\Tony\Desktop\Python Scripts\ML Practice\Computer V‭ision\YOLO Take Two\yolov3.weights',
                    help = 'path to pretrained weights')

    ap.add_argument('-c1', '--classes',
                    default=r'C:\Users\Tony\Desktop\Python Scripts\ML Practice\Computer V‭ision\YOLO Take Two\yolov3.txt',
                    help = 'path to text file containing class names')

    return ap.parse_args()


def get_output_layers(model):
    # does this need it's own function? it's just two lines...
    layer_names = model.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in model.getUnconnectedOutLayers()]
    # multiple output layers, need to get all of them
    
    return output_layers


def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    
    label = str(classes[class_id])
    color = colors[class_id]

    cv2.rectangle(img, (x,y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    # make label into a tuple of (lable, confidence)? how to draw confidence?


def generate_model(whts_dir, cfg_dir):
    # just delete this function entirely?
    model = cv2.dnn.readNet(whts_dir, cfg_dir)
    
    return model


def preprocess_image(image, img_dim, scale):
    # move some stuff from process image here?
    # shape image to certain values?
    
    width = image.shape[1]
    height = image.shape[0]
    # save these to unpad later?
    pad_color = [128,128,128]

    x_add = max(img_dim - width, 0) // 2  # if these aren't ints, might cause issue...
    y_add = max(img_dim - height, 0) // 2

    image = cv2.copyMakeBorder(image, y_add, y_add, x_add, x_add,
                                 cv2.BORDER_CONSTANT, value=pad_color)

    # is changing image size even necessary? cv2 seems pretty on top of stuff...

    # shape image to certain sizes?

    width = image.shape[1]
    height = image.shape[0]

    features = cv2.dnn.blobFromImage(image, scale, (img_dim,img_dim), (0,0,0),
                                     True, crop=False)

    return features  #, (width, height)


def unprocess_image():
    # remove padding and such?
    pass


def process_image(image, conf_threshold, nms_threshold, scale, img_dim, model):
    # takes around half a second, how to speed this up?
        # using gpu requires an entirely new opencv compile...
    # allow intaking height and width?

    width = image.shape[1]
    height = image.shape[0]

    demo_input = cv2.dnn.blobFromImage(image, scale, (img_dim,img_dim), (0,0,0),
                                       True, crop=False)
    # image, scale of image??, 416 is pixel count of image i think, base colors
    # true is for using cuda?

    model.setInput(demo_input)

    start = time.time()
    
    outputs = model.forward(get_output_layers(model))
    # this makes sure that all output layers get found

    end = time.time()
    print(end - start)

    class_ids = []
    confidences = []
    boxes = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > conf_threshold:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)

                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = center_x - w/2
                y = center_y - h/2

                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])


    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    for i in indices:
        i = i[0]
        box = boxes[i]

        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]

        draw_bounding_box(image, class_ids[i], confidences[i], round(x), round(y),
                          round(x+w), round(y+h))
        # draw confidence as well?

    return image


def process_batch(batch_dir, conf_threshold, nms_threshold, scale, img_dim, model):

    images = []
    
    for file in os.listdir(batch_dir):
        if file.endswith('.jpg') or file.endswith('.png'):
            img = cv2.imread((batch_dir + '/' + file))
            img = process_image(img, conf_threshold, nms_threshold, scale, img_dim, model)
            images.append(img)

    return images


def process_video(video_dir, conf_threshold, nms_threshold, scale, img_dim, model):
    # break video into separate images
        # use cv2.VideoCapture on path
        # use cv2.read on object created above
    # process each image
        # feed into process_image?
        # rewrite process_image here?
    # reassemble video
        # new video = cv2.VideoWriter()....keep using new video.write
        # need:
        # filename
        # fourcc        encoding type for video
        # fps
        # framseSize
        # is color?
        # demo: cv2.VideoWriter('video.avi',-1,1,(width,height))

    video = cv2.VideoCapture(video_dir)
    success, image = video.read()

    save_dir = str('second_processed_' + video_dir)
    fps = video.get(cv2.CAP_PROP_FPS)
    print(fps)
    
    width = image.shape[1]
    height = image.shape[0]

    processed_video = cv2.VideoWriter(save_dir,-1,fps,(width,height))
    # why is output so much bigger?

##    new_image = process_image(image, conf_threshold, nms_threshold, scale, img_dim, model)
##    processed_video.write(new_image)
    
    while success:
        # process image
        # write image to processed_video
        # get new success, image from video.read
        new_image = process_image(image, conf_threshold, nms_threshold, scale, img_dim, model)
        processed_video.write(new_image)
        success, image = video.read()

    # save video?? or is it automatically saved?
    video.release()
    processed_video.release()
    cv2.destroyAllWindows()


def real_time_video():
    # get most recent frame from video input
        # take most recent, even if in between frames haven't been processed
    # process frame
    # animate frame
        # framerate will just be limited by processing
    pass


if __name__ == '__main__':

    # setting up the hyperparameters

    weights_dir = 'yolov3.weights'
    config_dir = 'yolov3.cfg'
    classes_dir = 'yolov3.txt'

    image_dir = 'demo_img.png'

    batch_dir = 'demo_batch'

    video_dir = 'demo_video.MOV'

    image_dim = 64  # images are compressed to this pixel count for height and width

    conf_thresh = 0.4
    nms_thresh = 0.5
    value_scale = 0.00392  # scales the values in preprocessing

    show = False
    save = False

    # read class names
    classes = None
    with open(classes_dir, 'r') as class_file:
        classes = [line.strip() for line in class_file.readlines()]
    
    # generate different colors for different classes
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    
    network = cv2.dnn.readNet(weights_dir, config_dir)

    ########################################################################################

    # running something

##    demo_img = process_image(image_dir, conf_thresh, nms_thresh, value_scale, image_dim, network)
##    # img_dir, conf, nms, model
##    if show:
##        cv2.imshow('object detection: ', demo_img)
##        cv2.waitKey()

    demo = process_video(video_dir, conf_thresh, nms_thresh, value_scale, image_dim, network)

##    for demo_img in demo:
##        cv2.imshow('object detection: ', demo_img)
##        cv2.waitKey()

    if save:
        cv2.imwrite('demo_img_detection.jpg', demo_img)
    
    cv2.destroyAllWindows()





