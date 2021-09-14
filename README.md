# ✨ Model inputs & outputs flow
![](lux.png)
<!-- Gather the most resources and survive the night! -->

<!-- ![](https://github.com/Lux-AI-Challenge/Lux-Design-2021/raw/master/assets/daynightshift.gif) -->

# ✨ kaggle kernels
-  [Notebook 1 : Train model on many scenarios of an agent who follow deterministic strategy](https://www.kaggle.com/aithammadiabdellatif/lux-ai-dl-learn-the-strategy)

-  [Notebook 2 : Enhance the model using reinforcement learning](https://www.kaggle.com/aithammadiabdellatif/keras-lux-ai-reinforcement-learning)

# ✨ Approach

In notebook 1 we tackle the following steps:
- [x] Making a deterministic agent play many episodes and save them. ([Agent source code](https://github.com/aaiit/lux-AI/tree/main/Game))
- [x] Define data generator
- [ ] Radom fliping augmentation
- [x] Define custom loss 
- [x] Define the model. ([Model visualisation](https://viscom.net2vis.uni-ulm.de/PmSEAEjLx3yjq0qpBFwtadHHtdOJ36qpycXGp0y3LfXOBtjqiF))
- [x] Train the model.

In notebook 2
- [ ] Enhance the model using reinforcement learning.

# ✨ Challenges
- The map take different shapes
- Maintaining exploration 

# ✨ Loss function
The loss is based on MSE to be suited to game task
<!--  \sqrt{ \frac{1}{N_{units} }\sum_{i \in units } Y^{units}_i - \bar{Y^{units}_i}} + \sqrt{ \frac{1}{N_{cities} }\sum_{i \in cities} Y^{cities}_i -\bar{Y^{cities}_i}} -->

<a href="https://www.codecogs.com/eqnedit.php?latex=\sqrt{&space;\frac{1}{N_{units}&space;}\sum_{i&space;\in&space;units&space;}&space;Y^{units}_i&space;-&space;\bar{Y^{units}_i}}&space;&plus;&space;\sqrt{&space;\frac{1}{N_{cities}&space;}\sum_{i&space;\in&space;cities}&space;Y^{cities}_i&space;-\bar{Y^{cities}_i}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\sqrt{&space;\frac{1}{N_{units}&space;}\sum_{i&space;\in&space;units&space;}&space;Y^{units}_i&space;-&space;\bar{Y^{units}_i}}&space;&plus;&space;\sqrt{&space;\frac{1}{N_{cities}&space;}\sum_{i&space;\in&space;cities}&space;Y^{cities}_i&space;-\bar{Y^{cities}_i}}" title="\sqrt{ \frac{1}{N_{units} }\sum_{i \in units } Y^{units}_i - \bar{Y^{units}_i}} + \sqrt{ \frac{1}{N_{cities} }\sum_{i \in cities} Y^{cities}_i -\bar{Y^{cities}_i}}" /></a>
#### keras implimentation
```python
def custom_mean_squared_error(y_true, y_pred):
    y_units_true = y_true[:,:,:,:6]
    y_cities_true = y_true[:,:,:,6:]

    y_units_pred = y_pred[:,:,:,:6]
    y_cities_pred = y_pred[:,:,:,6:]
    
    is_unit = tf.keras.backend.max(y_units_true,axis = -1)
    is_city = tf.keras.backend.max(y_cities_true,axis = -1)
    
    y_units_pred*= K.stack([is_unit]*6, axis=-1)
    y_cities_pred*= K.stack([is_city]*2, axis=-1)
    
    loss1 = K.square(y_units_pred - y_units_true)/K.sum(is_unit)
    loss2 = K.square(y_cities_pred - y_cities_true)/K.sum(is_city)
    return K.concatenate([loss1,loss2])
```
<!-- ![](images/deepQlearning.png) -->

# ✨ Links
- [Lux AI Kaggle competition](https://www.kaggle.com/c/lux-ai-2021) 
- [Reinforcement Learning: An Introduction
 (MIT Press, Cambridge, MA, 2018)
](http://incompleteideas.net/book/the-book-2nd.html)

