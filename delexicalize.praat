#   This script was adapted to be used with Python3 by Alexandre Barroso (2024)
#   But the original script belongs to...
#   PRAAT SCRIPT: PURR2004 1.0
#   (Petra Wagner, 2004)
#
#   This script delexicalises speech files based on the PURR philosophy described in
#   Sonntag, Gerit (1998). Evaluation von Prosodie. Shaker Verlag: Aachen. This means
#   that both amplitude AND fundamental frequency information is kept intact in the
#   delexicalised signal. Also, unlike some other methods, no artificial "vowel"
#   is imitated by simply introducing a "formant" around the area of the second harmonic
#   Usage: select a Sound object you want to delexicalise and run script.

form Handle the command line
    sentence inputFilePath
    sentence outputFilePath
endform

# Read the sound file
Read from file... 'inputFilePath$'
sound = selected("Sound")

# Existing processing
intensity = To Intensity... 100 0
select sound
pitch = To Pitch... 0 75 500
meanPitch = Get mean... 0 0 Hertz
doublePitch = meanPitch * 2
hum = To Sound (hum)
select intensity
intensityTier = Down to IntensityTier
plus hum
hum_int = Multiply
hum_filt = Filter (one formant)... doublePitch doublePitch/2
Rename... 'outputFilePath$'

# Save the processed file
select hum_filt
Write to WAV file... 'outputFilePath$'
Remove
