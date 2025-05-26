![Preview](static/logo.png)  


---


<div align="center">
  <h2>Model capable of identifying all levels of Robtop (Geometry Dash :p) 🤖</h2> 
  <img src="static/robtop.png" alt="Preview" />
</div>

---

## EN🇬🇧 
### Development process  
A few months ago, as I began exploring the field of artificial intelligence, I wondered whether it would be possible to develop a model capable of recognizing Geometry Dash levels. My goal was not to exploit this for unfair advantage in Sparky, especially considering the evident hardware limitations—even when utilizing cloud resources. Ultimately, I decided to base the project on the pre-trained MobileNetV2 model from Keras, leveraging convolutional neural networks for image recognition. The complete code can be found in the repository [GDLvlDetector](https://github.com/ANGELUSD11/GDLvlDetector/). Initially, I attempted to build the network from scratch using TensorFlow, which surprisingly performed quite well in recognizing around two to five levels at most. However, the primary challenge with this approach was the sheer volume of data required and the hardware constraints. In the following sections, I will explain each phase of the training process I undertook.  

--

