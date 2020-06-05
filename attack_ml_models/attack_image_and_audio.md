# Attack image and audio

## Attack on Images
### One Pixel Attack
General attack may require that the noise can't exceed a certain size.  
$$\begin{array}{|c|c|c|c|}
\hline
 {0} & {42} & {30} & {32} \\
\hline
 {12} & {12} & {11} & {0} \\
\hline
 {32} & {0} & {3} & {3} \\
\hline
 {9} & {23} & {22} & {0} \\
\hline 
\end{array}$$

One pixel attack requires that only one pixel can be changed. 
$$\begin{array}{|c|c|c|c|}
\hline
 {0} & {0} & {0} & {0} \\
\hline
 {0} & {0} & {50} & {0} \\
\hline
 {0} & {0} & {0} & {0} \\
\hline
 {0} & {0} & {0} & {0} \\
\hline 
\end{array}$$ 

|Type|General Attack|One Pixel Attack|
|--|--|--|
|$\underset{e(x)^*}{\max}$|$f_{adv}(x + e(x))$|$f_{adv}(x + e(x))$|
|Subject to|$\left\|e(x)\right\| \leq L$ or L-infinity|$\left\|e(x)\right\|_0 \leq d, d=1$|

- $x$ : n-dimensional inputs, $x = (x_1,\cdots,x_n)$
- $f$ : image classifier (softmax output of model)
- $f_t(x)$ : the probability of class t clasified by the model given the input $x$
- $e(x)$ : additive adversarial perturbation according to $x$

**Untargeted Attack**  
$\underset{e(x)^*}{\argmin f_t(x+e(x))}$ where $t$ is the original class of $x$  

**Target attack**  
$\underset{e(x)^*}{\argmax f_{adv}(x+e(x))}$ where $adv$ is the targeted class

**NOTE** Both of the attack are subject to $\left\|e(x)\right\|_0 = 1$

## How do we find the exact pixel and value ?
### Do we really need the best perturbation ?
No! In fact, we only need to find a result that is sufficient to make the model be misclassified. It is not necessary to find the pixel and pixel value that make the model loss maximum.

## Differential Evolution  
During each iteration another set of candidata solutions(children) is generated according to the current population(parents). The the children are compared with their corresponding parents, surviving if they are more fitted (possess higher fitness value) than their parents. In such a way, only comparing the parent and his child, the goal of keeping diversity and improving fitness values can be simultaneously achieved.

**Advantages** :  
- High probability of finding global optima
  - due to diversity keeping mechainsms and the use of a set of candidate solutions (With randomness,it may encounter global optima)
- Require less information from target system
  - No need to calculate gradient, so it doesn't require too much details about model.
  - Independent of the classifier used

## Step of Differential Evolution
- Initialize candidates  
  - Now there is an $f(x)$, which squares each component of vector $x$ and adds up to average  
   Goal: Find a vector $x$ so that $f(x)$ is as small as possible    
   $f(x) = \frac{1}{4}\sum x_i^2 , x = \lbrace x_1,x_2,x_3,x_4 \rbrace , -5 \leq x_i \leq 5$  
  - At the beginning, randomly generated 10 sets of 4-dimensional vectors, each number is between 0 and 1  
  ```python
  pop = np.random.rand(popsize,dimensions)
  ```
  - Project these 10 sets of vectors between [-5, 5] and throw them into $f$ to calculate the value  

- Select candidates and generate  
  - First use the first vector as the target vector, then a candidate will be generated to see if the value of this candidate after $f$ is smaller  
  ```python
  # target vector can't be choose
  selected = np.random.choice(idxs,3,replace=False)
  mutant = selected[0] + mut * (selected[1] - selected[2] )
  # make sure the candidate between 0 and 1
  candidate = np.clip(mutant,0,1)
  ```
  - Note that we will not keep all the numbers in the candidate. How can we decide which values of the target vector will be replaced by the values of the corresponding positions of the candidate?
  - Crossp is a number between 0 and 1, random is a 4-dimensional vector, and records whether the obtained 4 numbers are less than crossp  
  For the target vector, if the cross_points corresponds to False, the value of the position of the target vector is retained, otherwise, the value of the position of the target vector is replaced by the value of the position of the mutant vector. This operation is the latest candidate
  ```python
  cross_points = np.random.rand(dimensions) < crossp
  # if True, replace the value of target vector by the value of candidate vector of corresponds position 
  trial = np.where(cross_points,mutant,pop[j]) # our new target vector
  ```

- Test candidates and substitue  
  Project the new candidate just obtained into the interval [-5, 5] and bring it into $f$ to get a value. If this value is smaller than the value brought into $f$ by the original target vector, replace the original target with this new candidate vector. If the trail vector is worse than the target vector, so the target vector is preserved and the trial vector is discarded.    
  In the next round, change a target vector and continue to do the same thing
  
## How is Differential Evolution used on one pixel attack?
Actuallly, the desired vector is a five element tuple $(x,y,R,G,B)$ ,which is produced by Differential Evolution   
$x,y$ coordinates : the $x$ and $y$ coordinates of the pixel to attack

**NOTE** Under the condition of the same iteration times and number of candidates, the larger the image is, the lower the success rate of the attack will be.

## Attack on Audio
- Attacked on ASR   
  Attack automatic speech recognition to make the transcript wrong
- Attacks on ASV  
  The adversary wants to craft an adversarial sample from a voice uttered by some source speaker, so that it is classified as one of the enrolled speakers (untargeted attack)
- Hidden Voice Attack

Both of attacking on ASR and ASV are based on trained model.

### Hidden Voice Attack  
The main job of signal processing is to extract a piece of important information from audio  
Purpose: Change the original audio, but the result of the changed audio after signal processing should not be too different from the original normal audio after signal processing  
Unlike the common attack, the general attack is to attack the model inference, which make it classify wrong. What this hidden voice attack does is to do something bad during signal preprocess 

## Progress of Audio Signal Processing
- Processing  
  Low Pass Filter & Noise Removal: Allow low-frequency signals to pass, but attenuate (or reduce) the passage of signals with frequencies higher than the cut-off frequency
- Signal Processing    
  - FFT : convert the time domain to the frequency domain, the purpose is to know what the component of this signal is, that is, how big is the component of a certain frequency, for example: the frequency of the sound emitted by people is generally up to 4KHz, so we can use this method  to analyze which audio zone most of the human voice is in.
  - Magnitude
  - Mel Filtering
  - Log of Powers
  - DCT  
  After obtaining the spectrogram, in order to get the sound features of appropriate size, it will then pass through a series of mel filter banks, which the filters at low frequencies are dense, and the filters at high frequencies are sparse.  
  After doing some mathematical operations, you will get the acoustic feature.
- Model Inference  
  Trained with neural network 

### Perturbation of Hidden Voice Attack
Goal : Origin Audio $\rightarrow$ Perturbation Audio
- Time Domain Inversion(TDI)
  - Use the many-to-one property of mFFT(FFT)
  - Two completely different signals in the time domain can have similar spectra. Such as the mfft results of $sin$ and $-sin$ wave  are the same
  - Modifying the audio in the time domain while preserving its spectrum, by inverting the windowed signal
  - inverting small windows across the entire signal, removes the smoothness
- Random Phase Generation(RPG)
  - For each frequency in the spectrum, the FFT returns the value in complex form $a+bi$, where $a$ and $b$ define the phase of a signal
  - magnitude $\sqrt{a^2 + b^2}$
  - Adjust $a$ and $b$ so that the magnitude still remains the same, the phase is different from the original (it doesn't sound like the original sound), but the magnitude spectrum will still remain the same
- High Frequency Addition(HFA)
  - In the process of signal processing, the low-pass filter will filter out the frequency bands which are much higher than the human voice to increase the accuracy of the VPS (Voice Processing System).
  - Add high frequencies to the audio that are filtered out during the preprocessing stage. The idea is to add a bunch of things that will be filtered out.  
  - Create high frequency $sin$ waves and add it to the real audio. Anyway, it will be filtered out and will not affect the results after preprocess.
  - If the sine waves have enough intensity, it has the potential to mask the underlying audio command to the human ear. To add a bunch of high-frequency $sin$ waves, hope to cover up the original audio
- Time Scaling(TS)
  - Fast forward the audio so that the model can correctly identify but people don’t understand what they are saying
  - Compress the audio in the time domain by discarding unnecessary samples and maintain the same sample rate
  - The audio is shorter in time, but retains the same spectrum as the origin  
  Waves are made up of several data points, sample rate is the number of data points per second.Shorten the time and remove some samples to ensure that the sample rate is the same







