import cv2
import numpy as np
import math
import autopy

def make_grid(name, *src):
    dst = np.zeros((height/2*3 , width/2*4), np.uint8)
    added = 0
    for var in src:
        x1 = (added%4)*width/2
        y1 = math.floor(added/4)*height/2
        x2 = x1 + width/2
        y2 = y1 + height/2
        dst[y1:y2, x1:x2] = cv2.resize(var, (width/2 , height/2))
        added += 1
    cv2.imshow(name, dst)

def init_list(list_name, times, value):
    for i in range(times):
        list_name.append(value)

def update_list(list_name, value):
    list_name.insert(0, value)
    list_name.pop()

width = 640
height = 480
kernel_param_1 = 3
kernel_param_2 = 3
canny_param_1 = 25
canny_param_2 = 50
canny_param_3 = 50
mouse_enabled = False
screen_width, screen_height = autopy.screen.get_size()
center_short = []
center_long = []
conv = []
conv2 = []

init_list(center_short, 10, (320,240))
init_list(center_long, 20, (320,240))
init_list(conv, 6, 6)
init_list(conv2, 10, 6)
cap = cv2.VideoCapture(1) #otwarcie kamery nr 1
cap.set(3,width) #ustawienie szerokosci przechwytywanego obrazu
cap.set(4,height) #ustawienie wysokosci przechwytywanego obrazu
while cap.isOpened(): #operacje wykonywane dla kazdej przechwyconej klatki
    try:
        ret, img = cap.read() #pobranie obrazu z kamery
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #konwersja do odcieni szarosci  
        #cv2.imshow('Etap 1 - Konwersja do odcieni szarosci', cv2.resize(grey, (grey.shape[1]/2 , grey.shape[0]/2)))
        canny = cv2.Canny( grey, canny_param_1, canny_param_2, canny_param_3); #ekstrakcja krawedzi za pomoca algorytmu Cannyego
        #cv2.imshow('Etap 2 - Wykrywanie konturow algorytmem Canny\'ego', cv2.resize(canny, (canny.shape[1]/2 , canny.shape[0]/2))) 
        kernel = np.ones((kernel_param_1, kernel_param_2),np.uint8)
        dilated = cv2.dilate(canny,kernel,iterations = 2)
        #cv2.imshow('Etap 3 - Dylatacja', cv2.resize(dilated, (dilated.shape[1]/2 , dilated.shape[0]/2)))
        eroded = cv2.erode(dilated,kernel,iterations = 2)
        
        #cv2.imshow('Etap 4 - Erozja', cv2.resize(eroded, (eroded.shape[1]/2 , eroded.shape[0]/2)))
        _, contours, _ = cv2.findContours(eroded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #znajdowanie krawedzi
        shapes = [] #lista na wszystkie ksztalty            
        for contour in contours:
          ar = cv2.arcLength(contour, 0)
          shapes.append(ar)
        if len(shapes) > 0:
            max_len = max(shapes)
            max_len_index = shapes.index(max_len) #indeks elementu o najwiekszym ksztalcie
            biggest_shape = contours[max_len_index] #najwiekszy ksztalt
            hull = cv2.convexHull(biggest_shape)
            #biggest_shape = cv2.approxPolyDP(biggest_shape,0.003*cv2.arcLength(biggest_shape,True),True)
            pic = np.zeros(eroded.shape, np.uint8)
            cv2.drawContours(pic, [biggest_shape], 0, (254, 254, 254), 2, 0)
            cv2.drawContours(pic, [biggest_shape], 0, (255, 255, 0), -1, 0)
            exp = np.zeros(eroded.shape, np.uint8)
            cv2.drawContours(exp, [biggest_shape], 0, (1, 1, 1), 3, 0)
            cv2.drawContours(exp, [biggest_shape], 0, (1, 1, 1), -1, 0)
            largeMask = np.zeros((exp.shape[0]+2, exp.shape[1]+2), np.uint8)
            cv2.floodFill(exp, largeMask, (0,0), (255,255,255));
            exp = cv2.bitwise_not(exp)
            _, contours2, _ = cv2.findContours(exp.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #znajdowanie krawedzi
            contours2[0] = cv2.approxPolyDP(contours2[0],0.005*cv2.arcLength(contours2[0],True),True)
            hull2 = cv2.convexHull(contours2[0])
            #print cv2.arcLength(hull2,True)
            exp2 = np.zeros(eroded.shape, np.uint8)
            exp3 = np.zeros(eroded.shape, np.uint8)
            cv2.drawContours(exp3, [contours2[0]], 0, (255, 255, 255), 1, 0)
            cv2.drawContours(exp3, [hull2], 0, (255, 255, 255), 1, 0)        
            cv2.drawContours(exp2, [contours2[0]], 0, (255, 255, 255), 2, 0)
            cv2.drawContours(exp2, [hull2], 0, (255, 255, 255), 1, 0)        
            hull2 = cv2.convexHull(contours2[0], returnPoints=False)
            defects2 = cv2.convexityDefects(contours2[0], hull2)
            #for defect in defects2:
                #cv2.circle(exp3, defect, 3, 255, -1)
                #print defect
            count_defects2 = 0
            points = []
            for i in range(defects2.shape[0]):
                s, e, f, d = defects2[i, 0]
                start = tuple(contours2[0][s][0])
                end = tuple(contours2[0][e][0])
                far = tuple(contours2[0][f][0])
                a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57
                if angle <= 90:
                    count_defects2 += 1
                    #cv2.circle(img, far, 2, [255, 0, 0], -1)
                    cv2.circle(exp3, far, 3, 255, -1)
                    points.append(far)
                cv2.line(exp3, start, end, [0, 255, 0], 2)
            
            update_list(conv2, len(points))
            #conv2.insert(0, len(points))
            #conv2.pop()
            #print points
            #cv2.imshow('Etap 4 - Wydobycie najdluzszej krawedzi', cv2.resize(pic, (pic.shape[1]/2 , pic.shape[0]/2)))
            bounding_rect_a, bounding_rect_b, bounding_rect_c, bounding_rect_d = cv2.boundingRect(biggest_shape)
            cv2.rectangle(img, (bounding_rect_a, bounding_rect_b), (bounding_rect_a + bounding_rect_c, bounding_rect_b + bounding_rect_d), (255, 100, 100), 0)
            #hull = cv2.convexHull(biggest_shape)
            cv2.drawContours(img, [biggest_shape], 0, (0, 0, 255), 3)
            cv2.drawContours(img, [hull], 0, (0, 0, 255), 0)
            approx = cv2.approxPolyDP(hull,0.012*cv2.arcLength(hull,True),True)              
            #drawing = np.zeros(img.shape, np.uint8)
            pic2 = np.zeros(pic.shape, np.uint8)
            cv2.drawContours(pic2, [biggest_shape], 0, 255, 0)
            pic3 = np.zeros(pic2.shape, np.uint8)
            cv2.drawContours(pic3, [biggest_shape], 0, 255, 0)
            cv2.drawContours(pic3, [hull], 0, 255, 0)
            pic4 = pic3.copy()
            fingertips = approx[np.where(approx[:, 0][:,1] <= 400)]
            #cv2.drawContours(img, [fingertips], 0, (0, 255, 255), 5)
            x_vert = []
            y_vert = []
            for fingertip in fingertips:
                cv2.circle(img, (fingertip[0,0],fingertip[0,1]), 5, [0, 255, 255], -1)
                cv2.circle(pic4, (fingertip[0,0],fingertip[0,1]), 6, 255, -1)
                x_vert.append(fingertip[0,0])
                y_vert.append(fingertip[0,1])
            if (len(x_vert)>0 and len(y_vert)>0):
                center_long_averages = [sum(col) / float(len(col)) for col in zip(*center_long)]
                cv2.circle(img, (int(center_long_averages[0]), int(center_long_averages[1])), 12, [255, 255, 255], 3)
                center_short_averages = [sum(col) / float(len(col)) for col in zip(*center_short)]
                cv2.circle(img, (int(center_short_averages[0]), int(center_short_averages[1])), 9, [255, 195, 195], 3)            
                hand_pos = (sum(x_vert)/float(len(x_vert)), sum(y_vert)/float(len(y_vert)))
                cv2.circle(img, (int(hand_pos[0]), int(hand_pos[1])), 6, [255, 100, 100], -1)
                delta = (center_long_averages[0] - center_short_averages[0], center_long_averages[1] - center_short_averages[1])
                #print delta
                if mouse_enabled:
                    if delta[0] > 2.2:
                        mouse_pos = autopy.mouse.get_pos()
                        if ((mouse_pos[0] +1*int(delta[0]) > 0) and (mouse_pos[0] +1*int(delta[0]) < screen_width)):
                            autopy.mouse.smooth_move(mouse_pos[0] +1*int(delta[0]), mouse_pos[1])
                            #print "prawo"
                    if delta[1] > 2.2:
                        mouse_pos = autopy.mouse.get_pos()
                        if ((mouse_pos[1] -1*int(delta[1]) > 0) and (mouse_pos[1] -1*int(delta[1]) < screen_height)):
                            autopy.mouse.smooth_move(mouse_pos[0], mouse_pos[1] -1*int(delta[1]))
                            #print "gora"
                    if delta[0] < -2.2:
                        mouse_pos = autopy.mouse.get_pos()
                        if ((mouse_pos[0] +1*int(delta[0]) > 0) and (mouse_pos[0] +1*int(delta[0]) < screen_width)):
                            autopy.mouse.smooth_move(mouse_pos[0] +1*int(delta[0]), mouse_pos[1])
                            #print "lewo"
                    if delta[1] < -2.2:
                        mouse_pos = autopy.mouse.get_pos()
                        if ((mouse_pos[1] -1*int(delta[1]) > 0) and (mouse_pos[1] -1*int(delta[1]) < screen_height)):
                            autopy.mouse.smooth_move(mouse_pos[0], mouse_pos[1] -1*int(delta[1]))
                            #print "dol"
                update_list(center_long, hand_pos)
                update_list(center_short, hand_pos)        
                #cv2.drawContours(drawing, [biggest_shape], 0, (0, 255, 0), 0)
                #cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 0)
                hull = cv2.convexHull(biggest_shape, returnPoints=False)
                defects = cv2.convexityDefects(biggest_shape, hull)
                count_defects = 0
                fars = []
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    start = tuple(biggest_shape[s][0])
                    end = tuple(biggest_shape[e][0])
                    far = tuple(biggest_shape[f][0])
                    a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                    b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                    c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                    angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57
                    if angle <= 90:
                        count_defects += 1
                        cv2.circle(img, far, 2, [255, 0, 0], -1)
                        cv2.circle(pic4, far, 3, 255, -1)
                        fars.append(far)
                    cv2.line(img, start, end, [0, 255, 0], 2)
                #print len(fars)
                conv.insert(0, count_defects)
                conv.pop()
                #cv2.imshow('Etap 10 - Rozpoznawanie konturow', cv2.resize(drawing, (drawing.shape[1]/2 , drawing.shape[0]/2)))
                #detected = np.zeros(pic.shape, np.uint8)
                #detected = img
                conv_avg = int(sum(conv)/len(conv))
                if conv_avg > 4 and mouse_enabled:
                    cv2.putText(img, "Released", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
                    autopy.mouse.toggle(False)
                elif conv_avg < 4 and mouse_enabled:            
                    cv2.putText(img, "Pressed", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
                    autopy.mouse.toggle(True)
                #cv2.putText(img, "Fingertips: " + str(len(fingertips)), (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
                #cv2.putText(img, "Convexity defects 1: " + str(conv_avg), (5, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
                #cv2.putText(img, "Convexity defects 2: " + str(int(sum(conv2)/len(conv2))), (5, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
                #cv2.imshow('Etap 11 - Efekt koncowy', cv2.resize(detected, (detected.shape[1]/2 , detected.shape[0]/2))) 
                cv2.putText(img, "Mouse enabled: " + str(mouse_enabled), (5, 460), cv2.FONT_HERSHEY_SIMPLEX, 1, 1) 
            cv2.imshow('Efekt koncowy', img)      
            make_grid('Demo', grey, canny, dilated, eroded, pic, pic2, pic3, pic4, exp, exp2, exp3)      
            #cv2.imshow('Krawedzie', drawing)     
        k = cv2.waitKey(10)
        if k == 27:
            break
        elif k == 32:
            mouse_enabled = not mouse_enabled 
    except:
        print( "Unexpected error")
        pass