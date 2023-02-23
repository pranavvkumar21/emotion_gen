# emotion_gen
This project aims to develop a computational model for generating personalities and emotions for robots using microbial fuel cells (MFCs) and external stimuli. The model builds on the personality generation model developed by Martin Grao and generates five personality traits based on the big 5 personality theory, including openness to experience, conscientiousness, extraversion, agreeableness, and neuroticism. Additionally, the model generates emotions based on Russell's two-dimensional theory of emotions, where emotions are described by their positivity (valence) and arousal (energy level).

The model uses MFC data along with external stimuli and the personality trait "neuroticism" to generate valence and arousal values, which are then used to calculate discrete emotions. The project tests the validity of the original model by testing it with MFC on different feeding cycles and comparing the expected and observed personality evolution. The updated model has also been tested to assess the impact of different parameters, such as MFC data and smoothing value, on the generated personality. The validity of generated emotions is checked by testing them against real-life scenarios and comparing the expected and observed results.

## Requirements
(-) Python 3.x
(-) Numpy
(-) Pandas
(-) Matplotlib

## Results
The personality evolution in the original model is validated as it follows the expected evolution pattern. The updated model also satisfies the expected evolution pattern and generates expected emotions most of the time. Out of the six tested scenarios, the emotion generation failed to generate one scenario, and the failure might have been caused due to mislabeling parts of the valence-arousal emotion space.

## Future Work
Future work includes improving the model's ability to generate emotions accurately and implementing the model on actual robots to test its real-world applicability.
