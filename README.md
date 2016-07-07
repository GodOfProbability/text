# Masking
This is a masking experiment that essentially deals with creating a mask over the characters in 
natural scenes. The data is generated synthetically and ground truth is developed using Otsu 
binarization. Different architectures are tried by varying depths etc.

The fundamental idea behind this Masking Project is that if you can train a CNN to mask the 
background that implies that you have trained it to focus on characters, without investing too
much in understanding the intrinsic quality of the characters. This network can further be used 
to train (through transfer learning) more sophisticated Text-non-Text classifier. 

In short this is the first step in building Text-non-Text classifier.

This work is done under a project that concerns [Text Detection](http://link.springer.com/chapter/10.1007%2F978-3-319-10593-2_34).


![Masking result from architecture_1](https://github.com/GodOfProbability/text/blob/master/architecture_1/results0.png)


TODO: The final heatmap is need to be formed. This map is essentially a saliency map that gives 
a map over any image correspodingly masking background from images. 

### License
BSD License.

