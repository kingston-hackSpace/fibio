#kingstonhack.space

#import the necessary module
import freenect
import cv2
import numpy as np

#modules for leds
import time
from rpi_ws281x import *
import argparse

#modules for random
from random import seed
from random import random

LED_COUNT = 356
LED_PIN = 12
LED_FREQ_HZ = 800000
LED_DMA = 10
LED_BRIGHTNESS = 200
LED_INVERT = False
LED_CHANNEL = 0

delay = 200

arm_1 = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]
arm_2 = [27,26,25,24,23,22,21,20,19,18,17,16,15,14]
arm_3 = [28,29,30,31,32,33,34,35,36,37,38,39,40,-1]
arm_4 = [53,52,51,50,49,48,47,46,45,44,43,42,41,-1]
arm_5 = [54,55,56,57,58,59,60,61,62,63,64,65,66,67]
arm_6 = [81,80,79,78,77,76,75,74,73,72,71,70,69,68]
arm_7 = [82,83,84,85,86,87,88,89,90,91,92,93,94,-1]
arm_8 = [107,106,105,104,103,102,101,100,99,98,97,96,95,-1]
arm_9 = [108,109,110,111,112,113,114,115,116,117,118,119,120,-1]
arm_10 = [134,133,132,131,130,129,128,127,126,125,124,123,122,121]
arm_11 = [135,136,137,138,139,140,141,142,143,144,145,146,147,148]
arm_12 = [162,161,160,159,158,157,156,155,154,153,152,151,150,149]
arm_13 = [163,164,165,166,167,168,169,170,171,172,173,174,175,176]
arm_14 = [189,188,187,186,185,184,183,182,181,180,179,178,177,-1]
arm_15 = [190,191,192,193,194,195,196,197,198,199,200,201,202,203]
arm_16 = [217,216,215,214,213,212,211,210,209,208,207,206,205,204]
arm_17 = [218,219,220,221,222,223,224,225,226,227,228,229,230,231]
arm_18 = [245,244,243,242,241,240,239,238,237,236,235,234,233,232]
arm_19 = [246,247,248,249,250,251,252,253,254,255,256,257,258,259]
arm_20 = [272,271,270,269,268,267,266,265,264,263,262,261,260,-1]
arm_21 = [273,274,275,276,277,278,279,280,281,282,283,284,285,286]
arm_22 = [300,299,298,297,296,295,294,293,292,291,290,289,288,287]
arm_23 = [301,302,303,304,305,306,307,308,309,310,311,312,313,314]
arm_24 = [328,327,326,325,324,323,322,321,320,319,318,317,316,315]
arm_25 = [329,330,331,332,333,334,335,336,337,338,339,340,341,-1]
arm_26 = [355,354,353,352,351,350,349,348,347,346,345,344,343,342]

counter_arm_1 = [1,355,331,327,305,297,279,268,253,239,227,209,201,180,175,151,149,-1]
counter_arm_2 = [28,2,354,332,326,306,296,280,267,254,238,228,208,202,179,176,150,-1]
counter_arm_3 = [29,27,3,353,333,325,307,295,281,266,255,237,229,207,203,178,177,-1]
counter_arm_4 = [54,30,26,4,352,334,324,308,294,282,265,256,236,230,206,204,-1,-1]
counter_arm_5 = [82,56,53,31,25,5,351,335,323,309,293,283,264,257,235,231,205,-1]
counter_arm_6 = [83,81,57,52,32,24,6,350,336,322,310,292,284,263,258,234,232,-1]
counter_arm_7 = [108,84,80,58,51,33,23,7,349,337,321,311,291,285,262,259,233,-1]
counter_arm_8 = [135,109,107,85,79,59,50,34,22,8,348,338,320,312,290,286,261,260,-1]
counter_arm_9 = [136,134,110,106,86,78,60,49,35,21,9,347,339,319,313,289,287]
counter_arm_10 = [163,137,133,111,105,87,77,61,48,36,20,10,346,340,318,314,288,-1]
counter_arm_11 = [164,162,138,132,112,104,88,76,62,47,37,19,11,345,341,317,315,-1]
counter_arm_12 = [191,190,165,161,139,131,113,103,89,75,63,46,38,18,12,344,342,316,-1]
counter_arm_13 = [218,192,189,166,160,140,130,114,102,90,74,64,45,39,17,13,343]
counter_arm_14 = [219,217,193,188,167,159,141,129,115,101,91,73,65,44,40,16,14,-1]
counter_arm_15 = [246,220,216,194,187,168,158,142,128,116,100,92,72,66,43,41,15,-1]
counter_arm_16 = [247,245,221,215,195,186,169,157,143,127,117,99,93,71,67,42,-1,-1]
counter_arm_17 = [274,273,248,244,222,214,196,185,170,156,144,126,118,98,94,70,68,-1]
counter_arm_18 = [301,275,272,249,243,223,213,197,184,171,155,145,125,119,97,95,69,-1]
counter_arm_19 =[302,300,276,271,250,242,224,212,198,183,172,154,146,124,120,96,-1,-1]
counter_arm_20 =[329,303,299,277,270,251,241,225,211,199,182,173,153,147,123,121,-1,-1]
counter_arm_21 =[356,330,328,304,298,278,269,252,240,226,210,200,181,174,152,148,122,-1]



x_values= [
0.475,0.527,0.568,0.598,0.62,0.634,0.643,0.648,0.648,0.646,0.642,0.636,0.629,0.621,0.624,0.635,0.645,0.656,0.666,0.675,0.682,0.686,0.687,0.684,0.675,0.659,0.635,0.6,0.714,0.731,0.739,0.74,0.736,0.728,0.717,0.704,0.69,0.676,0.661,0.647,0.633,0.626,0.641,0.658,0.676,0.695,0.714,0.734,0.753,0.771,0.788,0.8,0.809,0.811,0.899,0.885,0.865,0.842,0.816,0.79,0.763,0.737,0.712,0.689,0.667,0.647,0.629,0.613,0.595,0.611,0.629,0.65,0.673,0.699,0.727,0.758,0.79,0.825,0.86,0.896,0.932,0.965,0.998,0.95,0.902,0.857,0.814,0.774,0.737,0.704,0.675,0.649,0.626,0.606,0.589,0.565,0.58,0.597,0.618,0.643,0.672,0.705,0.743,0.785,0.832,0.883,0.939,0.999,0.969,0.903,0.842,0.788,0.741,0.699,0.663,0.632,0.606,0.584,0.566,0.552,0.54,0.516,0.524,0.535,0.548,0.566,0.588,0.614,0.647,0.685,0.73,0.783,0.843,0.912,0.989,0.909,0.832,0.766,0.71,0.662,0.622,0.59,0.563,0.542,0.526,0.514,0.505,0.499,0.495,0.478,0.477,0.479,0.482,0.489,0.499,0.513,0.532,0.557,0.589,0.629,0.678,0.737,0.807,0.692,0.633,0.584,0.546,0.516,0.494,0.477,0.467,0.46,0.457,0.456,0.458,0.461,0.465,0.451,0.443,0.437,0.432,0.428,0.427,0.43,0.436,0.448,0.466,0.492,0.527,0.573,0.498,0.457,0.428,0.408,0.395,0.389,0.389,0.392,0.398,0.406,0.415,0.425,0.436,0.447,0.449,0.436,0.422,0.407,0.393,0.379,0.365,0.353,0.344,0.338,0.336,0.34,0.352,0.374,0.267,0.265,0.271,0.281,0.296,0.313,0.332,0.352,0.371,0.39,0.409,0.426,0.442,0.457,0.47,0.455,0.438,0.419,0.398,0.375,0.351,0.326,0.299,0.273,0.247,0.222,0.201,0.183,0.127,0.162,0.197,0.233,0.268,0.302,0.334,0.363,0.39,0.415,0.437,0.456,0.473,0.487,0.494,0.478,0.46,0.439,0.414,0.387,0.355,0.321,0.282,0.241,0.197,0.15,0.102,0.045,0.107,0.165,0.219,0.268,0.313,0.353,0.388,0.419,0.446,0.468,0.488,0.504,0.517,0.54,0.53,0.517,0.501,0.481,0.458,0.429,0.396,0.357,0.312,0.261,0.204,0.141,0.071,0.126,0.199,0.264,0.32,0.369,0.411,0.446,0.476,0.5,0.519,0.535,0.547,0.556,0.562,0.582,0.579,0.574,0.567,0.557,0.542,0.524,0.5,0.471,0.435,0.391,0.339,0.278,0.207,0.305,0.37,0.424,0.468,0.504,0.532,0.554,0.571,0.583,0.591,0.596,0.598,0.598,0.613,0.617,0.619,0.62,0.618,0.613,0.604,0.591,0.572,0.546,0.512,0.469,0.415,0.349
]


y_values= [
0.057,0.108,0.157,0.204,0.247,0.286,0.322,0.354,0.382,0.407,0.428,0.446,0.461,0.474,0.495,0.485,0.472,0.457,0.437,0.415,0.388,0.357,0.321,0.281,0.235,0.184,0.129,0.068,0.108,0.173,0.231,0.283,0.328,0.367,0.4,0.428,0.452,0.471,0.487,0.499,0.509,0.531,0.525,0.517,0.505,0.49,0.472,0.448,0.419,0.384,0.343,0.294,0.237,0.172,0.184,0.255,0.316,0.368,0.411,0.446,0.475,0.497,0.515,0.528,0.537,0.544,0.548,0.55,0.564,0.566,0.567,0.566,0.562,0.555,0.544,0.529,0.508,0.481,0.447,0.404,0.351,0.287,0.4,0.452,0.494,0.526,0.55,0.567,0.579,0.586,0.589,0.59,0.588,0.584,0.58,0.587,0.596,0.604,0.611,0.616,0.62,0.621,0.619,0.612,0.599,0.579,0.552,0.514,0.621,0.643,0.656,0.662,0.663,0.66,0.653,0.644,0.634,0.623,0.612,0.6,0.588,0.584,0.597,0.612,0.627,0.642,0.658,0.674,0.688,0.701,0.711,0.718,0.72,0.715,0.702,0.796,0.79,0.779,0.763,0.744,0.724,0.703,0.681,0.661,0.641,0.622,0.604,0.588,0.574,0.56,0.574,0.591,0.609,0.63,0.652,0.677,0.703,0.731,0.759,0.788,0.816,0.843,0.866,0.908,0.869,0.83,0.792,0.756,0.722,0.69,0.661,0.634,0.611,0.59,0.572,0.556,0.542,0.535,0.549,0.565,0.585,0.608,0.635,0.665,0.698,0.736,0.777,0.822,0.87,0.92,0.967,0.904,0.846,0.792,0.744,0.701,0.663,0.63,0.601,0.576,0.555,0.538,0.524,0.512,0.49,0.498,0.509,0.523,0.541,0.562,0.588,0.619,0.655,0.697,0.745,0.8,0.861,0.93,0.867,0.797,0.736,0.683,0.638,0.6,0.568,0.542,0.521,0.504,0.491,0.482,0.475,0.47,0.453,0.454,0.457,0.462,0.47,0.481,0.496,0.516,0.542,0.573,0.612,0.66,0.716,0.782,0.682,0.624,0.576,0.538,0.507,0.483,0.466,0.453,0.444,0.439,0.436,0.436,0.437,0.44,0.426,0.421,0.416,0.414,0.413,0.415,0.42,0.429,0.444,0.464,0.492,0.529,0.575,0.512,0.469,0.436,0.413,0.397,0.387,0.382,0.382,0.384,0.389,0.395,0.403,0.411,0.42,0.42,0.409,0.397,0.385,0.374,0.363,0.354,0.346,0.341,0.34,0.343,0.353,0.37,0.396,0.293,0.284,0.283,0.287,0.296,0.308,0.322,0.337,0.352,0.368,0.384,0.398,0.412,0.425,0.435,0.422,0.406,0.39,0.372,0.352,0.332,0.311,0.289,0.269,0.249,0.232,0.218,0.208,0.148,0.173,0.201,0.229,0.258,0.286,0.313,0.338,0.362,0.384,0.403,0.421,0.436,0.454,0.439,0.422,0.403,0.381,0.356,0.328,0.297,0.264,0.229,0.192,0.153,0.114,0.076
]

light_list = [1]
light_timer = [10]

#arms = [arm_1,arm_2,arm_3,arm_4,arm_5,arm_6,arm_7,arm_8,arm_9,arm_10,arm_11,arm_12,arm_13,arm_14,arm_15,arm_16,arm_17,arm_18,arm_19,arm_20,arm_21,arm_22,arm_23,arm_24,arm_25,arm_26]

targ = 125
seed(1)
is_red = 0;
is_green = 0;

activity_timer = 25
counter = 0

direction = 0

clock_counter = 1
iris_counter =  13
spiral_timer =25
arm_spiral_counter = 20

saver_timer =1000

countdown_to_saver = 100
saver_state =0

def change_all(strip, color, offset):
		for i in range(offset,offset+5):
			strip.setPixelColor(i, color)
			strip.show()

def change_arm(strip, color):
		for i in range(14):
				if arm_1[i] > 0:
					strip.setPixelColor(arm_1[i],color)	
			


def change_arm_multi(strip):
		for i in range(26):
				arm_change(strip, Color(int(255*random()),int(255*random()),int(255*random())),i+1)	

def change_counter_arm_multi(strip):
		for i in range(21):
				counter_arm_change(strip, Color(int(255*random()),int(255*random()),int(255*random())),i+1)	

def arm_spiral(strip):
		global arm_spiral_counter
		local_counter = 0
		if (arm_spiral_counter>-1):
			arm_spiral_counter-=1
		else:
			arm_spiral_counter=20
		if arm_spiral_counter<10:
			local_counter = 1
		else:
			local_counter = 0
		
		if local_counter==0:
				make_blank(strip)
				arm_change(strip, Color(255,255,0),1)
				arm_change(strip, Color(255,255,0),3)
				arm_change(strip, Color(255,255,0),5)
				arm_change(strip, Color(255,255,0),3)
				arm_change(strip, Color(255,255,0),7)
				arm_change(strip, Color(255,255,0),9)
				arm_change(strip, Color(255,255,0),11)
				arm_change(strip, Color(255,255,0),13)
				arm_change(strip, Color(255,255,0),15)
				arm_change(strip, Color(255,255,0),17)
				arm_change(strip, Color(255,255,0),19)
				arm_change(strip, Color(255,255,0),21)
				arm_change(strip, Color(255,255,0),23)
				arm_change(strip, Color(255,255,0),25)
				
				arm_change(strip, Color(255,185,23),2)
				arm_change(strip, Color(255,185,23),4)
				arm_change(strip, Color(255,185,23),6)
				arm_change(strip, Color(255,185,23),8)
				arm_change(strip, Color(255,185,23),10)
				arm_change(strip, Color(255,185,23),12)
				arm_change(strip, Color(255,185,23),14)
				arm_change(strip, Color(255,185,23),16)
				arm_change(strip, Color(255,185,23),18)
				arm_change(strip, Color(255,185,23),20)
				arm_change(strip, Color(255,185,23),22)
				arm_change(strip, Color(255,185,23),24)
				arm_change(strip, Color(255,185,23),26)
		if local_counter==1:
				make_blank(strip)
				arm_change(strip, Color(255,255,0),2)
				arm_change(strip, Color(255,255,0),4)
				arm_change(strip, Color(255,255,0),6)
				arm_change(strip, Color(255,255,0),8)
				arm_change(strip, Color(255,255,0),10)
				arm_change(strip, Color(255,255,0),12)
				arm_change(strip, Color(255,255,0),14)
				arm_change(strip, Color(255,255,0),16)
				arm_change(strip, Color(255,255,0),18)
				arm_change(strip, Color(255,255,0),20)
				arm_change(strip, Color(255,255,0),22)
				arm_change(strip, Color(255,255,0),24)
				arm_change(strip, Color(255,255,0),26)

				arm_change(strip, Color(255,185,23),1)
				arm_change(strip, Color(255,185,23),3)
				arm_change(strip, Color(255,185,23),5)
				arm_change(strip, Color(255,185,23),3)
				arm_change(strip, Color(255,185,23),7)
				arm_change(strip, Color(255,185,23),9)
				arm_change(strip, Color(255,185,23),11)
				arm_change(strip, Color(255,185,23),13)
				arm_change(strip, Color(255,185,23),15)
				arm_change(strip, Color(255,185,23),17)
				arm_change(strip, Color(255,185,23),19)
				arm_change(strip, Color(255,185,23),21)
				arm_change(strip, Color(255,185,23),23)
				arm_change(strip, Color(255,185,23),25)
				
					


def arm_change(strip,color,arm):
		for i in range(14):
			if(arm==1):
				strip.setPixelColor(arm_1[i], color)
#				strip.show()
			if(arm==2):
				strip.setPixelColor(arm_2[i], color)
#				strip.show()
			if(arm==3):
				strip.setPixelColor(arm_3[i], color)
#				strip.show()
			if(arm==4):
				strip.setPixelColor(arm_4[i], color)
#				strip.show()
			if(arm==5):
				strip.setPixelColor(arm_5[i], color)
#				strip.show()
			if(arm==6):
				strip.setPixelColor(arm_6[i], color)
#				strip.show()
			if(arm==7):
				strip.setPixelColor(arm_7[i], color)
#				strip.show()
			if(arm==8):
				strip.setPixelColor(arm_8[i], color)
#				strip.show()
			if(arm==9):
				strip.setPixelColor(arm_9[i], color)
#				strip.show()
			if(arm==10):
				strip.setPixelColor(arm_10[i], color)
#				strip.show()	
			if(arm==11):
				strip.setPixelColor(arm_11[i], color)
#				strip.show()
			if(arm==12):
				strip.setPixelColor(arm_12[i], color)
#				strip.show()	
			if(arm==13):
				strip.setPixelColor(arm_13[i], color)
#				strip.show()	
			if(arm==14):
				strip.setPixelColor(arm_14[i], color)
#				strip.show()	
			if(arm==15):
				strip.setPixelColor(arm_15[i], color)
#				strip.show()	
			if(arm==16):
				strip.setPixelColor(arm_16[i], color)
#				strip.show()	
			if(arm==17):
				strip.setPixelColor(arm_17[i], color)
#				strip.show()	
			if(arm==18):
				strip.setPixelColor(arm_18[i], color)
#				strip.show()	
			if(arm==19):
				strip.setPixelColor(arm_19[i], color)
#				strip.show()	
			if(arm==20):
				strip.setPixelColor(arm_20[i], color)
#				strip.show()	
			if(arm==21):
				strip.setPixelColor(arm_21[i], color)
#				strip.show()	
			if(arm==22):
				strip.setPixelColor(arm_22[i], color)
#				strip.show()	
			if(arm==23):
				strip.setPixelColor(arm_23[i], color)
#				strip.show()	
			if(arm==24):
				strip.setPixelColor(arm_24[i], color)
#				strip.show()	
			if(arm==25):
				strip.setPixelColor(arm_25[i], color)
#				strip.show()	
			if(arm==26):
				strip.setPixelColor(arm_26[i], color)
#				strip.show()




def counter_arm_change(strip,color,counter_arm):
		for i in range(17):
			if(counter_arm==1):
				strip.setPixelColor(counter_arm_1[i]-1, color)
#				strip.show()
			if(counter_arm==2):
				strip.setPixelColor(counter_arm_2[i]-1, color)
#				strip.show()
			if(counter_arm==3):
				strip.setPixelColor(counter_arm_3[i]-1, color)
#				strip.show()
			if(counter_arm==4):
				strip.setPixelColor(counter_arm_4[i]-1, color)
#				strip.show()
			if(counter_arm==5):
				strip.setPixelColor(counter_arm_5[i]-1, color)
#				strip.show()
			if(counter_arm==6):
				strip.setPixelColor(counter_arm_6[i]-1, color)
#				strip.show()
			if(counter_arm==7):
				strip.setPixelColor(counter_arm_7[i]-1, color)
#				strip.show()
			if(counter_arm==8):
				strip.setPixelColor(counter_arm_8[i]-1, color)
#				strip.show()
			if(counter_arm==9):
				strip.setPixelColor(counter_arm_9[i]-1, color)
#				strip.show()
			if(counter_arm==10):
				strip.setPixelColor(counter_arm_10[i]-1, color)
#				strip.show()
			if(counter_arm==11):
				strip.setPixelColor(counter_arm_11[i]-1, color)
#				strip.show()
			if(counter_arm==12):
				strip.setPixelColor(counter_arm_12[i]-1, color)
#				strip.show()
			if(counter_arm==13):
				strip.setPixelColor(counter_arm_13[i]-1, color)
#				strip.show()
			if(counter_arm==14):
				strip.setPixelColor(counter_arm_14[i]-1, color)
#				strip.show()
			if(counter_arm==15):
				strip.setPixelColor(counter_arm_15[i]-1, color)
#				strip.show()
			if(counter_arm==16):
				strip.setPixelColor(counter_arm_16[i]-1, color)
#				strip.show()
			if(counter_arm==17):
				strip.setPixelColor(counter_arm_17[i]-1, color)
#				strip.show()
			if(counter_arm==18):
				strip.setPixelColor(counter_arm_18[i]-1, color)
#				strip.show()
			if(counter_arm==19):
				strip.setPixelColor(counter_arm_19[i]-1, color)
#				strip.show()
			if(counter_arm==20):
				strip.setPixelColor(counter_arm_20[i]-1, color)
#				strip.show()
			if(counter_arm==21):
				strip.setPixelColor(counter_arm_21[i]-1, color)
#				strip.show()
	
			



def make_purple(strip):
		for i in range(strip.numPixels()):
			strip.setPixelColor(i, Color(61,0,185))
		strip.show()

def make_red(strip):
		for i in range(strip.numPixels()):
			strip.setPixelColor(i, Color(255,0,0))
		strip.show()	

def make_blank(strip):
		for i in range(strip.numPixels()):
			strip.setPixelColor(i, Color(0,0,0))
			

def make_green(strip):
		for i in range(strip.numPixels()):
			strip.setPixelColor(i, Color(0+int(50*random()),255-int(50*random()),0+int(50*random())))
		strip.show()

#function to get RGB image from Kinnect
def get_video():
	array,_ = freenect.sync_get_video()[::2,::2]
	array = cv2.cvtColor(array,cv2.COLOR_RGB2BGR)
	return array

#function to get depth
def get_depth():
#	array,_ = freenect.sync_get_depth()[int(x),int(y)]
		depth = freenect.sync_get_depth()[0]
		return depth.astype(np.uint8)
#		return depth
#	array = array.astype(np.uint8)
#	return array
def spiral():
	global spiral_timer	
	global activity_timer
	if activity_timer>0:
		activity_timer -=1
	else:
		activity_timer = 25
	if(activity_timer ==25):
	
		if spiral_timer ==0:
			make_blank(strip)
			arm_spiral(strip,0)

		if spiral_timer ==1:
			make_blank(strip)
			arm_spiral(strip,1)
#			change_counter_arm_multi(strip)
		strip.show()	
	if spiral_timer ==0:
		spiral_timer = 1
	if spiral_timer ==1:
		spiral_timer = 0


def clock():
	global clock_counter
	global activity_timer	
	if activity_timer>0:
		activity_timer -=1
	else:
		activity_timer = 5
	if(activity_timer ==5):

		make_blank(strip)
		make_purple(strip)	
		if(clock_counter)==1:
			counter_arm_change(strip,Color(255,0,255),21)	
			counter_arm_change(strip,Color(125,0,125),1)
		else:
			counter_arm_change(strip,Color(255,0,255),clock_counter+1)	
			counter_arm_change(strip,Color(125,0,125),clock_counter)	
	
		if(clock_counter==21):
			clock_counter = 1
		else:
			clock_counter+=1
		strip.show()


def iris_close(color):
		global iris_counter
		make_blank(strip)
		if iris_counter>-1:
			for i in range(iris_counter,14):			
				strip.setPixelColor(arm_1[i],color)
				strip.setPixelColor(arm_2[i],color)
				strip.setPixelColor(arm_3[i],color)
				strip.setPixelColor(arm_4[i],color)
				strip.setPixelColor(arm_5[i],color)
				strip.setPixelColor(arm_6[i],color)
				strip.setPixelColor(arm_7[i],color)
				strip.setPixelColor(arm_8[i],color)
				strip.setPixelColor(arm_9[i],color)
				strip.setPixelColor(arm_10[i],color)
				strip.setPixelColor(arm_11[i],color)
				strip.setPixelColor(arm_12[i],color)		
				strip.setPixelColor(arm_13[i],color)
				strip.setPixelColor(arm_14[i],color)
				strip.setPixelColor(arm_15[i],color)
				strip.setPixelColor(arm_16[i],color)	
				strip.setPixelColor(arm_17[i],color)	
				strip.setPixelColor(arm_18[i],color)
				strip.setPixelColor(arm_19[i],color)
				strip.setPixelColor(arm_20[i],color)
				strip.setPixelColor(arm_21[i],color)
				strip.setPixelColor(arm_22[i],color)
				strip.setPixelColor(arm_23[i],color)
				strip.setPixelColor(arm_24[i],color)
				strip.setPixelColor(arm_25[i],color)
				strip.setPixelColor(arm_26[i],color)
		


def iris_open(color):
	global iris_counter	
	make_blank(strip)
	if iris_counter>-1:
		for i in range(iris_counter,14):			
			strip.setPixelColor(arm_1[i],color)
			strip.setPixelColor(arm_2[i],color)
			strip.setPixelColor(arm_3[i],color)
			strip.setPixelColor(arm_4[i],color)
			strip.setPixelColor(arm_5[i],color)
			strip.setPixelColor(arm_6[i],color)
			strip.setPixelColor(arm_7[i],color)
			strip.setPixelColor(arm_8[i],color)
			strip.setPixelColor(arm_9[i],color)
			strip.setPixelColor(arm_10[i],color)
			strip.setPixelColor(arm_11[i],color)
			strip.setPixelColor(arm_12[i],color)		
			strip.setPixelColor(arm_13[i],color)
			strip.setPixelColor(arm_14[i],color)
			strip.setPixelColor(arm_15[i],color)
			strip.setPixelColor(arm_16[i],color)	
			strip.setPixelColor(arm_17[i],color)	
			strip.setPixelColor(arm_18[i],color)
			strip.setPixelColor(arm_19[i],color)
			strip.setPixelColor(arm_20[i],color)
			strip.setPixelColor(arm_21[i],color)
			strip.setPixelColor(arm_22[i],color)
			strip.setPixelColor(arm_23[i],color)
			strip.setPixelColor(arm_24[i],color)
			strip.setPixelColor(arm_25[i],color)
			strip.setPixelColor(arm_26[i],color)
		


def iris_anim():
	global direction
	global activity_timer
	global iris_counter	
	if activity_timer>0:
		activity_timer -=1
	else:
		activity_timer = 2
	if(activity_timer ==2):
		if direction ==0:
			if(iris_counter<14):
				iris_counter+=1
				iris_close(Color(255,0,255))
			else:
				direction =1
		if direction ==1:
			if(iris_counter>0):
				iris_counter-=1
				iris_open(Color(255,0,255))
			else:
				direction =0



def get_x():
	d = get_depth()
	total = 1
	count = 1
	for i in range(0,480,20):
		for j in range(0,640,20):
			if d[i,j]>125 and d[i,j]<135:
				total=total + j
				count=count + 1
	return int(total/count)



def get_y():
	d = get_depth()
	total =1
	count =1
	for i in range(0, 480 ,20):
		for j in range(0,640,20):
			if d[i,j]>125 and d[i,j]<135:
				total=total + i
				count= count +1
	return int(total/count)

def nu_light_and_fade():
	d = get_depth()
	near = 0
	for i in range(0,480,20):
		for j in range(0,640,20):
			if d[i,j]>130 and d[i,j]<150:
				near = find_nearest(j,i)
#				strip.setPixelColor(near,Color(255-int(50*random()),0+int(50*random()),255-int(50*random())))
#		strip.setPixelColor(near,Color(255,0,255))
		
	if near is not 0:	
		if(len(light_list)<250):
			light_list.append(near)
			light_timer.append(10)
		else:
			light_list.pop(0)
			light_timer.pop(0)
			light_list.append(near)
			light_timer.append(10)
	
		
	for i in range(0,len(light_list)):
		light_timer[i]-=1
		if light_timer[i]<=0:
			strip.setPixelColor(light_list[i],Color(0,255,0))
#			light_list.pop(i)
#			light_timer.pop(i)
		else:
			if light_timer[i]<=10 :
				strip.setPixelColor(light_list[i],Color(light_timer[i]*25,255-(light_timer[i]*25),light_timer[i]*25))
		
	strip.show()



def find_nearest(x_pos,y_pos):
	nu_x_pos =0.0
	nu_y_pos =0.0 
	if x_pos>2:
		nu_x_pos = (float(x_pos)/640.0)
	if y_pos>2:
		nu_y_pos = 1.0-(float(y_pos)/480.0)
	x_diff = 0.0
	y_diff = 0.0
	record = 10.0
	winner = 0
	len = 0.0
	for i in range(0, 356):
		x_diff = x_values[i]-nu_x_pos
		y_diff = y_values[i]-nu_y_pos
		len = np.sqrt((x_diff*x_diff)+(y_diff*y_diff))
		if len<record:
			record = len
			winner = i
#		print(nu_x_pos,nu_y_pos,x_diff,y_diff,len)	
	return winner		
	

def close_proximity(x_pos,y_pos):
	d = get_depth()
	if x_pos==0 and y_pos==0:
		return 1;
	else:
		return 0;



def wheel(pos):
    """Generate rainbow colors across 0-255 positions."""
    if pos < 85:
        return Color(pos * 3, 255 - pos * 3, 0)
    elif pos < 170:
        pos -= 85
        return Color(255 - pos * 3, 0, pos * 3)
    else:
        pos -= 170
        return Color(0, pos * 3, 255 - pos * 3)

def rainbow(strip, wait_ms=20, iterations=1):
    """Draw rainbow that fades across all pixels at once."""
    for j in range(256*iterations):
        for i in range(strip.numPixels()):
            strip.setPixelColor(i, wheel((i+j) & 255))
        strip.show()
        time.sleep(wait_ms/1000.0)

def rainbowCycle(strip, wait_ms=20, iterations=1):
    """Draw rainbow that uniformly distributes itself across all pixels."""
    for j in range(256*iterations):
        for i in range(strip.numPixels()):
            strip.setPixelColor(i, wheel((int(i * 256 / strip.numPixels()) + j) & 255))
        strip.show()
        time.sleep(wait_ms/1000.0)

def randomcolours():
	for i in range (355):
		strip.setPixelColor(i,Color(0,int(255*random()),int(255*random())))
	strip.show()

def light_and_fade(x_pos,y_pos):
		near = find_nearest(x_pos,y_pos)
#		print(x_values[near])
#		print(y_values[near])
#		fade[near]-=1
#		strip.setPixelColor(near,Color(51*fade[near],255-(5*fade[near]),0))
		strip.setPixelColor(near,Color(255-int(50*random()),0+int(50*random()),255-int(50*random())))
#		strip.setPixelColor(near,Color(255,0,255))
		
		if(len(light_list)<150):
			light_list.append(near)
			light_timer.append(10)
		else:
			light_list.pop(0)
			light_timer.pop(0)
			light_list.append(near)
			light_timer.append(10)
		for i in range(0,len(light_list)):
			light_timer[i]-=1
			if light_timer[i]<=0:
				strip.setPixelColor(light_list[i],Color(0,255,0))
#				light_list.pop(i)
#				light_timer.pop(i)
			else:
				if light_timer[i]<10 :
					strip.setPixelColor(light_list[i],Color(light_timer[i]*25,255-(light_timer[i]*25),light_timer[i]*25))
		strip.show()
#		for i in range(0,365):
#			if fade[i]<5:
#				fade[i]+=1
		

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-c', '--clear', action='store_true', help= 'clear the display on exite')
	args = parser.parse_args()
	strip = Adafruit_NeoPixel(LED_COUNT, LED_PIN, LED_FREQ_HZ, LED_DMA, LED_INVERT, LED_BRIGHTNESS, LED_CHANNEL)
	strip.begin()
#	make_green(strip)

	saved_x_pos = -1
	saved_y_pos = -1	
	while 1:
		#get a frame from Camera
#		frame = get_video()		#get a frame from depth sensor
#1 is RHS 500 is LHS
		d=get_depth()
		
		x_pos = get_x()
		y_pos = get_y()

#		print(x_pos,y_pos,countdown_to_saver)	
		light_and_fade(x_pos,y_pos)
#		arm_spiral(strip)
		if abs(x_pos-saved_x_pos)<75 and abs(y_pos-saved_y_pos)<75:
			countdown_to_saver-=1
		else:
			saver_state = 1
			saved_x_pos = x_pos
			saved_y_pos = y_pos
			countdown_to_saver = 100

		if countdown_to_saver <0:
			saver_state = 0

		if saver_state == 0:
			is_green=0
			if saver_timer<-1:
				saver_timer=1000
			else:
				saver_timer-=1
			if saver_timer == 1000:
				make_blank(strip)
			if saver_timer<1000 and saver_timer>750:
				arm_spiral(strip)
			if saver_timer<750 and saver_timer>500:
				clock()
#			if saver_timer==500:
#				iris_counter=13
#			if saver_timer<500 and saver_timer>2:
#				iris_anim()
			if saver_timer==500:
				rainbowCycle(strip)
			if saver_timer<499 and saver_timer>250:
				randomcolours()			
			if saver_timer==249:
				saver_timer=1000
				
#		
		else:
			if  close_proximity(x_pos,y_pos) ==1:
				make_red(strip)	
				is_green=0
		
			else:
				if is_green == 0:
					make_green(strip)
					is_green = 1

				nu_light_and_fade()
		

		strip.show()
	
#spiral()
#		if activity_timer == 25:
#		clock()
		
		


#		if x_pos > 275:
#			if y_pos >120 and y_pos < 130:
#				arm_change(strip,Color(255,0,0),11)
#				arm_change(strip,Color(255,0,0),12)
#			if y_pos >130 and y_pos < 150:
#				arm_change(strip,Color(255,0,255),13)
#				arm_change(strip,Color(255,0,255),14)
#			if y_pos >150 and y_pos < 170:
#				arm_change(strip,Color(255,255,0),14)
#				arm_change(strip,Color(255,255,0),15)
#		if x_pos < 275 and x_pos > 150:
#			if y_pos >120 and y_pos < 130:
#				arm_change(strip,Color(0,255,0),16)
#				arm_change(strip,Color(255,0,0),17)
#			if y_pos >130 and y_pos < 150:
#				arm_change(strip,Color(0,255,255),18)
#				arm_change(strip,Color(0,255,255),19)
#			if y_pos >150 and y_pos < 170:
#				arm_change(strip,Color(0,255,255),20)
#				arm_change(strip,Color(0,255,0),21)
#		if x_pos < 150:
#			if y_pos >120 and y_pos < 130:
#				arm_change(strip,Color(0,255,0),4)
#				arm_change(strip,Color(255,0,0),5)
#			if y_pos >130 and y_pos < 150:
#				arm_change(strip,Color(255,255,255),6)
#				arm_change(strip,Color(255,255,0),7)
#			if y_pos >150 and y_pos < 170:
#				arm_change(strip,Color(0,255,255),8)
#				arm_change(strip,Color(255,255,0),9)
			
			

#			is_green = 0
#		change_all(strip, Color(targ, 0 , 255))
#		cv2.imshow('RGB image',frame)
		#diplay depth image
#		cv2.imshow('Depth image' ,depth)
		#quit if escape pressed
		k = cv2.waitKey(5) & 0xFF
		if k == 27:
			break
	cv2.destroyAllWindows()
