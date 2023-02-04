"""
@author Igor Romanica + Dimitrij Leshchenko
@version 1.0

Dies ist der Python-Code für unser Office Hero Spiel - ein Guitar Hero Ableger.
Wir nutzen OpenCV um Bilder der Webcam einzulesen und tracken den vom Spieler benutzen Textmarker,
mit dem er die Noten treffen soll, die eingeblendet werden.
Außerdem wird den Webcam-Bildern die Benutzeroberfläche und die Hinweise des Spiels eingeblendet.
Der Song wird aus einer Midi-Datei eingelesen. Es können Midi-Nachrichten empfangen werden, die Auskunft
über die Markerfarbe, den ausgwählten Songtitel und das Instrument geben. Außerdem ob die automatische Farb-
erkennung gestartet wurde oder der Song gestartet bzw. gestoppt wurde.
Da mehrere Operationen parallel und ohne Zeitverzörzgerung ablaufen, werden sie in Threads gestartet.
Zu den Threads gehören
-listenOnChange (lauscht nach Midi Nachrichten)
-playScreen (stellt die Noten und das Interface auf dem Screen dar)

Programm-Ablauf in dieser Python-Datei:
1. setzen der Standardwerte (Markerfarbe, Fenstergröße, Midi-Ports (erste Midi-Output/Input-Port der Liste)
2. starten der zwei Threads (listenOnChange, playScreen)
3. die Threads laufen endlos weiter im Loop (while 1).

Detailierte Beschreibung der Threads:
listenOnChange: Lauscht auf eingehende Midi-Nachrichten, setzt die Werte für Markerfarbe (jeweils die Unter- und
    Obergrenze der Farbwerte im HSV-Farbraum (l_h, u_h, l_s, u_s, l_v, u_v) / starten der Farberkennung (detect_color),
playScreen: Fügt die UI zu dem Webcam-Video hinzu, wie die Target Area (Bereich in dem Noten getroffen werden können),
    Rechteck um den erkannten Input-Marker

Lesen Sie bitte die Kommentare der jeweiligen Funktionen für weitere Details zu den Thread Methoden.
Schauen Sie sich für eine Übersicht des Zusammenspiels der Threads bitte das entsprechende Diagramm in der Doku an.

Verwendete Bibliotheken:
cv2 (OpenCV): Einlesen der Webcam-Bilder, UI-Gestaltung, Zeichnen der Noten-Symbole, Erkennung vom Input-Marker
mido (MIDI Objects for Python): Kommunikation zwischen Python- und JavaScript-/HTML-Welt, Einlesen der Midi-Datei.
threading: Ermöglicht die parallel laufenden Threads
NumPy: Wird für die Erstellung der Masken verwendet und die Kombination mit
    dem ursprünglichen Bild (detectMarker)

"""
import colorsys
import random
import time
import mido
from threading import Thread
import cv2
import numpy as np
import pygame

import timeit  # to check execution time (of detect_marker)
import optical_flow
import config # for global arrows for optical_flow
import struct
# prevMarkerPos: vorherige Markerposition
prevMarkerPos = []
prevMarker2Pos = []

# FX List
# 0 = optical flow (arrows)
# 1 = opt flow hsv
# 2 = opt flow warp  
# 3 = charcoal, 
# 4 = shade, 
# 5 = spread, 
# 6 = emboss, 
# 7 = edge, 
# 8 = blur
# 9 = motion blur
# 10 = colorspace
# 11 = Board drawer
# 12 = office hero
# list has: order no and bool = no. of order of effect in effect chain, bool = on/off
FXlist = [-1 for i in range(13)]

colorspaceList = ['bilevel','grayscale','palette','truecolor','colorseparation','optimize']
colorSpace = 'grayscale'

# Midi Ports
outport = mido.open_output(mido.get_output_names()[0])
inport = mido.open_input(mido.get_input_names()[0])

print("mido.get_output_names(): ", mido.get_output_names())
print("mido.get_input_names(): ", mido.get_input_names())

WIDTH = 640
HEIGHT = 480

marker_offscreen = True  # ist der Marker im sichtbaren Bereich?
marker2_offscreen = True  # ist der Marker2 im sichtbaren Bereich?

# Positionsdaten vom erkannten Marker-Bereich (rectangle)
marker_area_left_top = -1  # (x,y)
marker_area_right_bottom = -1  # (x,y)
# ig: zweiter Marker
marker2_area_left_top = -1  # (x,y)
marker2_area_right_bottom = -1  # (x,y)


# init (HSV) Farbwerte für Gelb
l_h, u_h = 25, 42
l_s, u_s = 150, 245
l_v, u_v = 50, 200

# ig: zweiter Marker
l2_h, u2_h = -1, -1
l2_s, u2_s = -1, -1
l2_v, u2_v = -1, -1


# Wird für die clock.tick(ms) Methode verwendet. ms = Zeit in Millisekunden, die ein Thread-Loop mindest braucht.
clock = pygame.time.Clock()

# um die Markerfarbe zu erkennen
detect_color = False
# ig: zweiter Marker
detect_color2 = False
detect_timer = False
color_red = False  # hilft dabei die Farbwinkel-Maske von rot zu erstellen
loop_timer = time.time()  # hilft bei der automatischen Farberkennung (detectColor)
loop_timer_delta = 0

# ig: paint_canvas (array) to be painted on with colors
paint_canvas = np.empty((480, 640, 3), dtype=np.uint8)
# ig: alpha_canvas is for the alpha blend - blend only this area! the colored spots are white here
alpha_canvas = np.empty((480, 640, 3), dtype=np.uint8)
reset_canvas = False  # um parallel das Canvas zu resetten
empty_canvas = True  # um zu checken ob das Canvas noch leer ist

## marker color for the painitng canvas (rgb)
marker_color = (0, 170, 255)
# ig: zweiter Marker
marker2_color = (0, 0, 0)


def playScreen():
    """ Erzeugt den Video-Output und prüft an entsprechenden stellen den Video-Input (detectMarker).
    """
    global detect_color, FXlist

    # src: https://stackoverflow.com/questions/52068277/change-frame-rate-in-opencv-3-4-2
    frame_rate = 30
    prev = 0
    vid = cv2.VideoCapture(0)
    # https://learnopencv.com/how-to-find-frame-rate-or-frames-per-second-fps-in-opencv-python-cpp/
    fps = vid.get(cv2.CAP_PROP_FPS)
    print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

    # for optical flow  src: https://stackoverflow.com/questions/43496397/extract-optical-flow-as-data-numbers-from-live-feed-webcam-using-python-open
    # arrows = []
    # cam = video.create_capture(fn)
    ret, prev = vid.read()
    prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    show_hsv = True
    show_glitch = False
    config.cur_glitch = prev.copy()

    while 1:
        time_elapsed = time.time() - prev
        start_time = timeit.default_timer()
        ret, frame = vid.read()
        frame = cv2.flip(frame, 1)  # spiegel das Bild
        # FX List
        # 0 = optical flow (arrows)
        # 1 = opt flow hsv
        # 2 = opt flow warp  
        # 3 = charcoal, 
        # 4 = shade, 
        # 5 = spread, 
        # 6 = emboss, 
        # 7 = edge, 
        # 8 = blur
        # 9 = motion blur
        # 10 = colorspace
        # 11 = board drawer (interactive installation attempt)
        # 12 = office hero (detect marker area smaller)
        # list has: order no and bool = no. of order of effect in effect chain, bool = on/off
        
        # first build list order:
        fx_order = [(FXlist[i], i) for i in range(len(FXlist))]
        fx_order = list(sorted(fx_order))
        for (order_no, fx_no) in fx_order:
            if order_no > -1:
                if fx_no == 0:  # opt flow arrows
                    frame, prevgray = optical_flow.applyOpticalFlow(frame, prevgray)
                elif fx_no == 1:  # opt flow hsv
                    frame, prevgray = optical_flow.applyOpticalFlow(frame, prevgray, True)
                elif fx_no == 2:  # opt flow warp  
                    frame, prevgray = optical_flow.applyOpticalFlow(frame, prevgray, False, True)
                elif fx_no == 3:
                    frame = charcoal(frame)
                elif fx_no == 4:
                    frame = shade(frame)
                elif fx_no == 5:
                    frame = spread(frame)
                elif fx_no == 6:
                    frame = emboss(frame)
                elif fx_no == 7:
                    frame = edge(frame)
                elif fx_no == 8:
                    frame = blur(frame)
                elif fx_no == 9:
                    frame = motionBlur(frame)
                elif fx_no == 10:
                    frame = colorspace(frame, colorSpace)
                elif fx_no == 11:
                    if detect_color or detect_color2:
                        frame = detectColor(frame) if detect_color else detectColor(frame, False)
                    else:
                        frame = detectMarker(frame)
                        # print("detectMarker time: ", timeit.default_timer() - start_time)
                        if u2_h != -1:  # detectMarker2 wenn zweiter Marker aktiviert wurde (aktuell durch detect_marker2)
                            frame = detectMarker(frame, False)
                
        sendExecTime(timeit.default_timer() - start_time)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # frame = applyRandomEffect(frame, 2)
        cv2.imshow('Office Hero', frame)
        cv2.setWindowProperty('Office Hero', cv2.WND_PROP_TOPMOST, 1)
        clock.tick(50)

def applyRandomEffect(frame, modeNo):
    if modeNo == 2:
        frame = colorspace(frame, colorSpace)
    elif modeNo == 3:
        frame = charcoal(frame)
    elif modeNo == 4:
        frame = shade(frame)
    elif modeNo == 5:
        frame = spread(frame)
    elif modeNo == 6:
        frame = emboss(frame)
    elif modeNo == 7:
        frame = edge(frame)
    elif modeNo == 8:
        frame = blur(frame)
    return frame
    

def detectMarker(frame, first_marker = True):
    """
     Zeichnet Rechteck um den Marker herum auf dem "frame"-Bild ein, falls er erkannt wurde und setzt die Positionsdaten.
     Gibt das Bild wieder zurück.

     first_marker == False bedeutet, dass der zweite Marker erkann wird

     Input: frame (Webcam-Bild mit eingezeichneter UI)
     Output: frame (Webcam-Bild mit eingezeichneter UI, mit Rechteck um Marker herum, falls er erkannt wurde),
        Marker-Positon wird global gesetzt (marker_area_left_top, marker_area_right_bottom)

     Funktion wird von playScreen aufgerufen.

     Details:
     Aktualisiert die Positionsdaten des Marker-Rechtecks (marker_area_left_top, marker_area_right_bottom).
     Der Marker wird anhand der Range der HSV-Farbwerte ermittelt. Dafür werden drei Masken erstellt jeweils für den
     Wertebereich (inRange()) von dem Farbwinkel (Hue), der Sättigung (Saturation) und der Helligkeit (Value).
     Falls eine rote Markerfarbe gewählt wurde, werden zwei Farbwinkel-Masken addiert. Grund hierfür ist, dass Rot im
     HSV-Farbraum einem Winkel von 0 Grad entspricht und Werte sowohl größer Null, als auch kleiner 360 Grad, umfasst.
     Die lower Boundary entspricht also einem Wert von unter 360 Grad (und > 340°) und die obere Grenzwert einem Wert
     über 0 Grad (und < 20°). Die drei Masken werden zu einer Maske multipliziert. Im Anschluss wird der untersuchte
     Bildausschnitt (roi_rectangle) mit der Maske kombiniert (bitwise_and()).
     Nun wird der Bildausschnitt nach Konturen untersucht und von den Konturen wird die größt erkannte Kontur ermittelt.
     Es wird ein Rechteck um den erkannten Markerbereich gezeichnet auf das Bild (frame).
     """
    global reset_canvas, paint_canvas, alpha_canvas, marker_area_left_top, marker_area_right_bottom, marker2_area_left_top, marker2_area_right_bottom, l_h, u_h, l_s, u_s, l_v, u_v, l2_h, u2_h, l2_s, u2_s, l2_v, u2_v, WIDTH, prevMarkerPos, prevMarker2Pos, marker_offscreen, marker2_offscreen

    # nehme nur den Ausschnitt vom rectangle als ROI
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, s, v = cv2.split(frame_hsv)

    # Rot entspricht einem Winkel von 0 Grad im HSV-Farbraum und umfasst Werte sowohl größer Null, als auch kleiner 360 Grad.
    # Die lower Boundary entspricht also einem Wert von unter 360 Grad (und > 340°) und die obere Grenzwert einem Wert über 0 Grad (und < 20°)
    # Im Falle einer roten Farbe werden einfach zwei Farbwinkel-Masken addiert.

    if not color_red:
        lower_h = np.array([l_h]) if first_marker else np.array([l2_h]) 
        upper_h = np.array([u_h]) if first_marker else np.array([u2_h])
        mask_h = cv2.inRange(h, lower_h, upper_h)
    else:
        lower_h = np.array([l_h]) if first_marker else np.array([l2_h])
        upper_h = np.array([u_h]) if first_marker else np.array([u2_h])
        lower_mask = cv2.inRange(h, np.array([0]), upper_h)
        upper_mask = cv2.inRange(h, lower_h, np.array([180]))
        mask_h = lower_mask + upper_mask

    lower_s = np.array([l_s]) if first_marker else np.array([l2_s])
    upper_s = np.array([u_s]) if first_marker else np.array([u2_s])
    mask_s = cv2.inRange(s, lower_s, upper_s)

    lower_v = np.array([l_v]) if first_marker else np.array([l2_v])
    upper_v = np.array([u_v]) if first_marker else np.array([u2_v])
    mask_v = cv2.inRange(v, lower_v, upper_v)

    # kombinieren der drei Binärmasken
    mask = mask_h * mask_s * mask_v
    mask_result = cv2.bitwise_and(frame_gray, frame_gray, mask=mask)

    # https://learnopencv.com/contour-detection-using-opencv-python-c/#Drawing-Contours-using-CHAIN_APPROX_NONE
    # contours, hierarchy = cv2.findContours(mask_result, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv2.findContours(mask_result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contours, hierarchy = cv2.findContours(mask_result, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    area_detected = []

    # Loope über die erkannten Konturen, füge Bereiche in die area_detected Liste hinzu, die groß genug sind.
    for c in contours:
       # Fläche für das Rechteck ermitteln --> min x,y und max x,y Werte = Rechteckfläche
       vals_x = []
       vals_y = []
       for dot in c:
           vals_x += [dot[0,0]]
           vals_y += [dot[0,1]]
       if(len(vals_x) > 0 and len(vals_y) > 0):
           min_x, max_x = min(vals_x), max(vals_x)
           min_y, max_y = min(vals_y), max(vals_y)

           # füge nur Bereiche hinzu, die die Mindestgröße erfüllen
           if min_x != max_x and min_y != max_y and abs(min_x - max_x) > 15 and abs(min_y - max_y) > 15:
               area_detected += [[[min_x, min_y], [max_x, max_y]]]

    # loope auf der Suche nach dem größten Bereich. Wir wollen den Index ermitteln.
    index = -1
    span_x = 0
    span_y = 0
    for i in range(len(area_detected)):
        [min_x, min_y], [max_x, max_y] = area_detected[i]
        if max_x - min_x > span_x and max_y - min_y > span_y:
            index = i
            span_x = min_x - max_x
            span_y = min_y - max_y

    # aktualisiere die Marker Area Positionsdaten, falls ein Bereich erkannt wurde. Male ein Rechteck um den Marker herum.
    if index != -1:
        [min_x, min_y], [max_x, max_y] = area_detected[index]
        frame = colorFx(frame, max_x, max_y)
        if first_marker:
            marker_area_left_top = (min_x, min_y)
            marker_area_right_bottom = (max_x, max_y)
            cv2.rectangle(frame, marker_area_left_top, marker_area_right_bottom, (10, 200, 0), 1)
            marker_offscreen = False
        else:
            marker2_area_left_top = (min_x, min_y)
            marker2_area_right_bottom = (max_x, max_y)
            cv2.rectangle(frame, marker2_area_left_top, marker2_area_right_bottom, (0, 100, 255), 1)
            marker2_offscreen = False

        if first_marker:
            prevMarkerPos = [marker_area_left_top, marker_area_right_bottom]
        else:
            prevMarker2Pos = [marker2_area_left_top, marker2_area_right_bottom]

        sendScreenPosFreq(max_x, max_y)  # to determine the synth freq

    else:
        if first_marker and marker_area_left_top == -1 and not marker_offscreen:  # marker war bereits zuvor offscreen!
            marker_offscreen = True
            sendScreenPosFreq(-1,-1)  # sendScreenPosFreq(-1,-1) --> sende per Midinachricht, dass kein Ton gespielt wird (-1 = marker 1)
        elif first_marker:
            marker_area_left_top = -1
            marker_area_right_bottom = -1
            prevMarkerPos = []
        else:  # marker2
            if marker_area_left_top == -1 and not marker_offscreen:  # marker war bereits zuvor offscreen!
                marker2_offscreen = True
                sendScreenPosFreq(-1,-1)  # sendScreenPosFreq(-2,-2) --> sende per Midinachricht, dass kein Ton gespielt wird (-2 = marker 2)
            else:
                marker2_area_left_top = -1
                marker2_area_right_bottom = -1
                prevMarker2Pos = []
    

    # todo: ig: paint canvas acording to the marker position
    
    if reset_canvas:
        paint_canvas[:] = (0,0,0)
        alpha_canvas[:] = (0,0,0)
        reset_canvas = False
    frame = paintCanvas(frame, prevMarkerPos) if first_marker else paintCanvas(frame, prevMarker2Pos, False)
    # ig: interactive installation: color fx on frame
    return frame

def sendExecTime(time_delta):
    midi_bytes = int_to_midi_bytes(int(time_delta*100))  # time_delta in centiseconds (seond * 100)
    # message = mido.Message('sysex', time=time_delta)
    message = mido.Message('sysex', data=midi_bytes)
    outport.send(message)


def sendScreenPosFreq(gotX, gotY):
    global WIDTH
    """ sendet die Position x mit Midi an JS
    """
    # send freq via midi outport
    # print("sendScreenPosFreq x: ", x)
    
    # IG: sende nur x Position und berechne die Frequenz in JS!
    if gotY >= 0:  # sende X Position nur, falls Marker erkann wurde [also gotY nicht -1 oder -2 ist]
        freq_msg_x = mido.Message('songpos', pos=gotX)
        outport.send(freq_msg_x)
        # IG: sende nur y Position als pitch und berechne die Lautstärke in JS!
    # sende ggf. per Midinachricht, dass kein Ton gespielt wird (gotY = -1 --> marker 1 | gotY = -2 --> marker 2)
    freq_msg_y = mido.Message('pitchwheel', pitch=gotY)
    outport.send(freq_msg_y)

#ig: new
def paintCanvas(image, marker_pos, first_marker = True):
    global paint_canvas, alpha_canvas, empty_canvas
    """
    mixes the painting Canvas with the orig. image and returning the mixed image 
    (orig. image with painting canvas overlay (0 - value = transparente maske)
    """
    if len(marker_pos) > 0:
        min_x, min_y = marker_pos[0]
        max_x, max_y = marker_pos[1]

        # 1. paint area of pixels in marker_pos with the color of the marker
        # replace subarray: https://medium.com/@alexppppp/replacing-part-of-2d-array-with-another-2d-array-in-numpy-c83144576ddc
        # example: Let’s replace elements in array a in the 4th & 5th rows and in the 3rd, 4th, 5th & 6th columns with array b:
        # a[3:5,2:6] = b
        color_canvas = np.zeros((max_y-min_y, max_x-min_x, 3), dtype=np.uint8)
        color_canvas[:] = list(marker_color) if first_marker else list(marker2_color)
        paint_canvas[min_y:max_y, min_x:max_x] = color_canvas # ig: Farbwert ermitteln
        alpha_canvas[min_y:max_y, min_x:max_x] = (155, 155, 155)
        if empty_canvas:  # sende Nachricht an JS per Midi, dass das Canvas nicht mehr leer ist (wieder resettet werden kann)
            # print("empty_canvas True")
            msg = mido.Message('note_on', note=50)  # die note ist nur ein dummy Wert
            outport.send(msg)
        empty_canvas = False

    # 2. mix das paintCanvas mit dem originalen Bild
    # image = cv2.addWeighted(src1=image,alpha=0.5,src2=paint_canvas,beta=0.5,gamma=0)
    # src: comment of https://stackoverflow.com/questions/72171816/how-to-speed-up-opencv-blending#comment127521161_72171816
    return np.where(alpha_canvas, paint_canvas, image)


# ig: new
def colorFx(frame, gotX, gotY):
    """
    colorEffect depending on the x-coordinate of the detected marker pointing area
    gets, "colors" and returns the frame
    left area: blue
    right area: red
    middle: normal
    """
    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # define range of blue color in HSV

    hue_val=int((gotX/WIDTH)*180)
    # print("hue_val: ", hue_val)
    # upper_blue = np.array([130,255,255])
    upper_blue = np.array([hue_val+20 if hue_val+20 < 179 else 179,255,255])
    # lower_blue = np.array([110,50,50])
    lower_blue = np.array([hue_val if hue_val+20 < 159 else 159,50,50])
    # Threshold the HSV image to get only blue colors
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    #set the bounds for the red hue
    # lower_red = np.array([160,100,50])

    h, s, v = cv2.split(hsv)

    # lower_h = np.array([150])
    # upper_h = np.array([49])
    # lower_mask = cv2.inRange(h, np.array([0]), upper_h)
    # upper_mask = cv2.inRange(h, lower_h, np.array([180]))
    # mask_red = lower_mask + upper_mask


    lower_red = np.array([0,50,50])
    upper_red = np.array([89,255,255])
    mask_red = cv2.inRange(hsv, lower_red, upper_red)

    # invert mask_blue
    mask_blue = cv2.bitwise_not(mask_blue)

    # invert mask_red
    mask_red = cv2.bitwise_not(mask_red)

    # Bitwise-AND mask_blue and original image
    res_blue = cv2.bitwise_and(frame,frame, mask = mask_blue)
    res_red = cv2.bitwise_and(frame,frame, mask = mask_red)

    # filter out black
    # res_red_white = np.add(res_red, np.where(res_blue == 0, 1, 0))
    whiteFactor = int((1-(gotY / HEIGHT))*255)
    
    res_red = np.add(res_red, np.where(res_blue == 0, 1, 0))
    res_red = np.add(res_red, np.where(res_red < 1, whiteFactor, 0))
    res_red = res_red.astype(dtype=np.uint8)

    res_blue = np.add(res_blue, np.where(res_red == 0, 1, 0))
    res_blue = np.add(res_blue, np.where(res_blue < 1, whiteFactor, 0))
    res_blue = res_blue.astype(dtype=np.uint8)

    # https://predictivehacks.com/blending-images/
    # Now, we can blend them, we need to define the weight (alpha) of the target image
    # as well as the weight of the filter image
    # in our case we choose 80% target and 20% filter
    
    # setze gotX in Relation zur Bildschirmposition
    # if gotX < width/2 --> blue
    # if gotX >= width/2 --> red

    # // IG: panning according to x val
    #    factor = ((x/WIDTH) - 0.5)*2

    factor = ((gotX/WIDTH) - 0.5)*2  # ig: test if works
    # print("factor red / blue: ", factor)
    blended = cv2.addWeighted(src1=frame,alpha=1-factor,src2=res_blue,beta=factor,gamma=0)
    # if factor < 0:
    #     # factor = 1 - (gotX / (WIDTH / 2))
    #     blended = cv2.addWeighted(src1=frame,alpha=1-factor,src2=res_blue,beta=factor,gamma=0)
    # else:
    #     # factor = (gotX - (WIDTH/2)) / (WIDTH/2)
    #     blended = cv2.addWeighted(src1=frame,alpha=1-factor,src2=res_red,beta=factor,gamma=0)
    return blended



def detectColor(img, first_marker = True):
    """ Farbeerkennung wurde gestartet (detect_color = True).
     Blendet 5 Sekunden lang das Rechteck ein, das den Bereich markiert in dem die Farbe erkannt wird
     und eine entsprechende Nachricht.
     Liest nach dieser Zeit den Farbbereich aus und setzt die untere und obere Farbwert-Grenzen mit einer zusätzlichen Toleranz.
     Wird aufgerufen von playScreen() falls die Farberkennung aktiviert wurde (detect_color = True).
     Farberkennung wird per Midi-Message aktiviert (listenOnChange() -> if msg.value==11 -> detect_color = True).
     Input: img (Webcam-Bild mit GUI)
     Output: img (Webcam-Bild mit GUI und neuen Elementen)
     Gibt das Bild (img) wieder zurück.
     first_marker == True bedeutet, dass die Farbe des ersten Markers erkann wird / False: Farbe des zweiten Markers wird erkannt
     """
    global detect_color, detect_color2, detect_timer, l_h, u_h, l_s, u_s, l_v, u_v, l2_h, u2_h, l2_s, u2_s, l2_v, u2_v, color_red, marker_color, marker2_color
    time_delta = time.time() - detect_timer  # Berechne wie lange die Farbekennung läuft (max. 5 sek)

    # Füge die UI Elemente hinzu
    cv2.rectangle(img, (WIDTH//2 - 155, HEIGHT//2 +50), (WIDTH//2 + 155, HEIGHT//2 +30), (0, 0, 0), -1)
    cv2.putText(img, "Place marker into rectangle in 5 sec.", (WIDTH//2 - 150, HEIGHT//2 +45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    cv2.rectangle(img, (WIDTH//2 - 10, HEIGHT//2 -30), (WIDTH//2 + 10, HEIGHT//2 -10), (0, 0, 0), -1)
    cv2.putText(img, str(int(6-time_delta)), (WIDTH//2 - 6, HEIGHT//2 -15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    # Bereich in dem die Farbe erkannt wird
    y1, y2 = HEIGHT//2 -145, HEIGHT//2 -135
    x1, x2 = WIDTH//2 - 5, WIDTH//2 + 5
    color_red = False

    if time_delta > 5:
        detect_timer = False
        detect_color = False
        detect_color2 = False

        # Teilbereich des Bildes erhalten (roi)
        roi = img[y1:y2, x1:x2]
        liH, liS, liV = [], [], []
        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        for i in range(len(roi)):
            for h,s,v in roi_hsv[i]:
                liH += [h]
                liS += [s]
                liV += [v]
        minH = min(liH); maxH = max(liH)
        minS = min(liS); maxS = max(liS)
        minV = min(liV); maxV = max(liV)

        # checke ob rote Farbe den 0° Winkel umspannt...
        if maxH > 160 and minH < 22:
            color_red = True
            maxH = max([x for x in liH if x < 22])
            minH = min([x for x in liH if x > 160])

        (h, s, v) = (maxH, maxS, maxV)

        # setze Farbe
        if first_marker:
            l_h, u_h = minH-8 if minH-8 > 0 else 0, maxH+8 if maxH+8 < 180 else 180
            l_s, u_s = minS-25 if minS-25 > 0 else 0, maxS+25 if maxS+25 < 255 else 255
            l_v, u_v = minV-25 if minV-25 > 0 else 0, maxV+25 if maxV+25 < 255 else 255

            marker_color = colorsys.hsv_to_rgb(h/180, s/255, v/255)
            marker_color = int(marker_color[0] * 255), int(marker_color[1] * 255), int(marker_color[2] * 255)
            # rgb to bgr --> reverse
            marker_color = tuple(reversed(marker_color))
            print("marker_color: ", marker_color)
        else:
            l2_h, u2_h = minH-8 if minH-8 > 0 else 0, maxH+8 if maxH+8 < 180 else 180
            l2_s, u2_s = minS-25 if minS-25 > 0 else 0, maxS+25 if maxS+25 < 255 else 255
            l2_v, u2_v = minV-25 if minV-25 > 0 else 0, maxV+25 if maxV+25 < 255 else 255

            marker2_color = colorsys.hsv_to_rgb(h/180, s/255, v/255)
            marker2_color = int(marker2_color[0] * 255), int(marker2_color[1] * 255), int(marker2_color[2] * 255)
            # rgb to bgr --> reverse
            marker2_color = tuple(reversed(marker2_color))
            print("marker2_color: ", marker2_color)

    return cv2.rectangle(img, (x1, y1), (x2, y2), (10, 200, 0), 2) if first_marker else cv2.rectangle(img, (x1, y1), (x2, y2), (0, 100, 255), 2)
# Convert an integer to a MIDI byte array
def int_to_midi_bytes(value):
    return struct.pack('>B', value & 0x7F)

# Example usage
# value = 100
# midi_bytes = int_to_midi_bytes(value)
# print(midi_bytes)

def listenOnChange():
    """ Prüft ob neue Midi Nachrichten (msg) eingetroffen sind sind und setzt die Variablenwerte entsprechend.
     Nachrichtenwerte:
     Markerfarbe (gelb, rot, blau, magenta), Farberkennung, Songauswahl und Start / Stop
     Läuft andauernd im Thread thread_listenOnChange.
     """
    global colorSpace, colorspaceList, FXlist, l_h, u_h, l_s, u_s, l_v, u_v, detect_color, detect_color2, detect_timer, color_red, reset_canvas, empty_canvas
    while 1:
        for msg in inport:
            if msg.type == 'control_change':
                ##### Textmarker Color #####
                ### Read if FX is turned on (msg.control==8)/off(msg.control==9) (toggle)
                if(msg.control==7):  # colorspace id wird übermittelt
                    # colorspaceList = ['bilevel','grayscale','palette','truecolor','colorseparation','optimize']
                    colorSpace = colorspaceList[msg.value]
                elif(msg.control==8):
                    FXlist[msg.value] = 0  # setze Order auf 0
                elif(msg.control==9):
                    FXlist[msg.value] = -1 # setze Order auf -1 (off)
                elif(msg.control==10 and msg.value in [1,2,3,4]):  # setze color_red auf False, falls eine Farbe gewählt wurde
                    color_red = False
                if(msg.control==10 and msg.value==1): #Gelb
                    l_h, u_h = 25, 42
                    l_s, u_s = 120, 255
                    l_v, u_v = 50, 248
                elif(msg.control==10 and msg.value==2): #Rot (magenta)
                    l_h, u_h = 180, 255
                    l_s, u_s = 120, 255
                    l_v, u_v = 50, 248
                elif(msg.control==10 and msg.value==3): #Blau
                    l_h, u_h = 100, 115
                    l_s, u_s = 120, 255
                    l_v, u_v = 50, 248
                elif(msg.control==10 and msg.value==4): #Grün
                    l_h, u_h = 55, 75
                    l_s, u_s = 120, 255
                    l_v, u_v = 50, 248
                elif(msg.control==10 and msg.value==11):  # Detect color
                    detect_color = True
                    detect_color2 = False
                    detect_timer = time.time()
                elif(msg.control==10 and msg.value==12):  # Detect color2
                    detect_color2 = True
                    detect_color = False
                    detect_timer = time.time()
                elif(msg.control==10 and msg.value==9):  ##### Clear Canvas #####
                    reset_canvas = True
                    empty_canvas = True
        clock.tick(50)

########## IMAGE WAND EFFECTS BELOW ######
##### Implosion / Explosion efffects #####
# https://stackoverflow.com/questions/64067196/pinch-bulge-distortion-using-python-opencv

from wand.image import Image
import numpy as np
import cv2

def implode(myImg, amount):
    with Image.from_array(myImg) as img:
        img.virtual_pixel = 'black'
        img.implode(amount)
        # img.save(filename='zelda1_implode.jpg')
        # convert to opencv/numpy array format
        img_implode_opencv = np.array(img)
        # img_implode_opencv = cv2.cvtColor(img_implode_opencv, cv2.COLOR_RGB2BGR)
    return img_implode_opencv


def colorspace(myImg, colorSpace):
    with Image.from_array(myImg) as img:
        # img.type = 'colorseparation'
        # img.type = random.choice(colorSpace)
        img.type = colorSpace
        img = np.array(img)
    return img


# display result with opencv
# cv2.imshow("IMPLODE", img_implode_opencv)
# cv2.imshow("EXPLODE", img_explode_opencv)
# cv2.waitKey(0)

def charcoal(myImg):
    # Read image using Image function
    with Image.from_array(myImg) as img:
        # Charcoal fx using charcoal() function
        img.charcoal(radius = 4, sigma = 0.65)
        # img.save(filename ="ch_koala_2.jpeg")
        img = np.array(img)
    return img

def shade(myImg):
    with Image.from_array(myImg) as img:
        img.shade(gray=True,
              azimuth=286.0,
              elevation=45.0)
        img = np.array(img)
    return img


def spread(myImg):
    with Image.from_array(myImg) as img:
        img.spread(radius=8.0)
        img = np.array(img)
    return img

def emboss(myImg):
    with Image.from_array(myImg) as img:
        img.transform_colorspace('gray')
        img.emboss(radius=3.0, sigma=1.75)
        img = np.array(img)
    return img

def edge(myImg):
    with Image.from_array(myImg) as img:
        img.transform_colorspace('gray')
        img.edge(radius=1)
        img = np.array(img)
    return img

def blur(myImg):
    with Image.from_array(myImg) as img:
        img.blur(radius=0, sigma=3)
        img = np.array(img)
    return img

##### MotionBlur ####
# todo: motion blur für eine bestimmte Farbmaske? (Hue-Maske)
# src: https://www.geeksforgeeks.org/opencv-motion-blur-in-python/
# loading library
def motionBlur(img):
    # Specify the kernel size.
    # The greater the size, the more the motion.
    kernel_size = 3000
    
    # Create the vertical kernel.
    kernel_v = np.zeros((kernel_size, kernel_size))
    
    # Create a copy of the same for creating the horizontal kernel.
    kernel_h = np.copy(kernel_v)
    
    # Fill the middle row with ones.
    kernel_v[:, int((kernel_size - 1)/2)] = np.ones(kernel_size)
    kernel_h[int((kernel_size - 1)/2), :] = np.ones(kernel_size)
    
    # Normalize.
    kernel_v /= kernel_size
    kernel_h /= kernel_size
    
    # Apply the vertical kernel.
    vertical_mb = cv2.filter2D(img, -1, kernel_v)
    
    # Apply the horizontal kernel.
    horizonal_mb = cv2.filter2D(img, -1, kernel_h)
    return img

def motionBlurHsv(img):
    # Convert image from BGR to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Get the hue channel
    hue = hsv[:, :, 0]

    # Specify the kernel size.
    # The greater the size, the more the motion.
    kernel_size = 30

    # Create the vertical kernel.
    kernel_v = np.zeros((kernel_size, kernel_size))

    # Create a copy of the same for creating the horizontal kernel.
    kernel_h = np.copy(kernel_v)

    # Fill the middle row with ones.
    kernel_v[:, int((kernel_size - 1)/2)] = np.ones(kernel_size)
    kernel_h[int((kernel_size - 1)/2), :] = np.ones(kernel_size)

    # Normalize.
    kernel_v /= kernel_size
    kernel_h /= kernel_size

    # Apply the vertical kernel to the hue channel.
    vertical_mb = cv2.filter2D(hue, -1, kernel_v)

    # Apply the horizontal kernel to the hue channel.
    horizonal_mb = cv2.filter2D(hue, -1, kernel_h)

    # Replace the hue channel in the HSV image with the motion blurred hue.
    hsv[:, :, 0] = vertical_mb

    # Convert image back to BGR color space
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img


##### end of motion blur #####


##### color hue switcher #####
def colorSwitcher(frame_no):
    """ switches the color of the color recognition to perform motion blur and other fx on the color. """ 
    global hue_up, hue_lo
    # hue_span = 90
    # hue_up = 150
    # hue_lo = hue_up - hue_span
    modulo_rate = 30  # 30 = every 1 second if 30 fps
    if (frame_no % modulo_rate) == 0:
        hue_up += (hue_up + 10) % 180
        hue_lo += (hue_lo + 10) % 180




##### Starte alle Threads #####
thread_listenOnChange = Thread(target=listenOnChange)
thread_playScreen = Thread(target=playScreen)

thread_listenOnChange.start()
thread_playScreen.start()
