## Image Classification with Classic and Deep Learning Techniques

Òscar Lorente Corominas

Ian Riera Smolinska Master in Computer Vision Universitat Autònoma de Barcelona

Aditya Sangram Singh Rana

08193 Bellaterra, Barcelona, Spain

{oscar.lorentec, ianpau.riera, adityasangramsingh.rana}@e-campus.uab.cat

Abstract -To classify images based on their content is one of the most studied topics in the field of computer vision. Nowadays, this problem can be addressed using modern techniques such as Convolutional Neural Networks (CNN), but over the years different classical methods have been developed. In this report, we implement an image classifier using both classic computer vision and deep learning techniques. Specifically, we study the performance of a Bag of Visual Words classifier using Support Vector Machines, a Multilayer Perceptron, an existing architecture named InceptionV3 and our own CNN, TinyNet, designed from scratch. We evaluate each of the cases in terms of accuracy and loss, and we obtain results that vary between 0.6 and 0.96 depending on the model and configuration used.

Keywords -Computer vision, Image classification, Bag of Visual Words, Support Vector Machines, Deep Learning, Multilayer Perceptron, Convolutional Neural Networks.

## I. INTRODUCTION

During the last years, there has been a great increase in the number of applications in which image classification is useful. Helping people organise their photo collections, analysing medical images or identifying what's around selfdriving cars are just a few examples. These tasks require precisely labeled large-scale datasets, and most of them include a huge variety of image types, from dogs or cats, to landscapes, roads, and so on.

In image classification, given an input image, the goal is to predict the class which it belongs to. This is not a big deal for humans, but teaching computers to see is a difficult problem that has become a broad area of research interest, and both classic computer vision and Deep Learning (DL) techniques have been developed. Classic techniques use local descriptors to try to find similarities between images, but today, advances in technology allow the use of DL techniques to automatically extract representative features and patterns from each image. However, to understand these concepts it is first necessary to review traditional techniques.

In this report, we present different image classification systems trained on a specific dataset detailed in Section III. We first explore the Bag of Visual Words (BoVW) approach, which consists on extracting local features from the images and clustering them to create visual words. For each image, its histogram of visual words is used as a global feature to classify it. On the other hand, we use a Multilayer Perceptron (MLP) as a first step towards DL. To improve the performance of the system, we use CNNs, which are

Fig. 1. Simplified scheme of the image classification system

<!-- image -->

more suitable for image classification. Specifically, we first refine an existing architecture, InceptionV3 [1], and finally design a CNN from scratch and analyze the impact of each layer, parameters, activation function, and more.

## II. PROBLEM DEFINITION

Given a dataset divided into 8 different classes, for each image in the dataset, the goal is to predict the class it belongs to. To do so, we implement and evaluate four different systems: BoVW approach, MLP based, and CNNs architectures: fine-tuning an existing one and designing one from scratch. In each case, the model is trained with a subset of images, and tested with unseen images to validate the performance by means of the accuracy and loss. In Fig. 1 we can see a simplified scheme of the system.

## III. DATA

The dataset contains 2688 images from 8 different classes: coast, forest, highway, inside\_city, mountain, open\_country, street and tall\_building. In Fig. 2 a sample image from each class is presented. To properly train and evaluate the implemented systems, the dataset is divided into a training set of 1881 images (70%) and a test set, that contains 807 images (30%).

Before start developing our system, it is important to analyze the dataset. For example, if the number of samples of each class is unevenly distributed (i.e. unbalanced dataset), using accuracy as the evaluation metric is not a good idea. In Fig. 3 it can be observed that the data samples are more or less equally distributed across the different classes, so the dataset is balanced.

<!-- image -->

Fig. 2. Image sample from class (a) coast, (b) forest, (c) highway, (d) inside\_city, (e) mountain, (f) open\_country, (g) street and (h) tall\_building

<!-- image -->

Fig. 3. Samples in each class for the train (blue) and test (orange) sets

<!-- image -->

## IV. BAG OF VISUAL WORDS

The Bag of Visual Words (BoVW) approach consists on, given some training data, extract some local descriptors, cluster them in the multidimensional feature space to create visual words and count the number of words each image has (i.e. histogram of visual words). Therefore, a histogram is generated for each labeled image, and used to train a classifier such as Support Vector Machines (SVM). A toy scheme is presented in Fig. 4. In this section, the methods used to implement the BoVW system are explained in detail, and the results obtained with each configuration are presented and analyzed. For that purpose, we compute the accuracy with 8 (stratified) fold cross-validation in all cases.

## A. Keypoints and descriptors

In the BoVW approach a feature detection algorithm is used to detect keypoints and extract local descriptors from each image, so the first step is to find which is the one that works best in our case using a k-Nearest Neighbors (KNN) classifier. The descriptors tested are: SIFT [2] (vanilla and dense), SURF [3] (vanilla and dense) and DAISY [4] (only dense).

- 1) Vanilla descriptors: In this scenario, keypoints are extracted using the detection algorithm of the corresponding local descriptor: SIFT or SURF (DAISY does not have any keypoint detector). An example of this detection is presented in Fig. 5b.
- 2) Dense descriptors: Instead of using the detection algorithm to extract keypoints, we create a grid of spatially equidistant keypoints, which is translated into a more dense representation of the image. This is shown in Fig. 5c.

Fig. 4. Toy scheme of the BoVW classification systemFig. 5. (a) Image sample, (b) keypoint detection and (c) generated dense keypoints

<!-- image -->

The results are presented in Tab. I. As observed, the dense descriptors outperform vanilla SIFT and SURF. The main reason is that vanilla SIFT and SURF depend on the performance of the keypoint detector: if this detection is poor, the classification fails. When using dense keypoints, we get a representative descriptor of the images even if the content of the image is plain (low textures) or has repetitive patterns. In this case, the best results are obtained with dense SIFT, so when the descriptor is mentioned in this paper, it will be referring to dense SIFT .

The hyperparameters used to create the dense keypoints are also fine-tuned, and the results show that the best performance is obtained when the number of keypoints is larger. On the other hand, vanilla SIFT is scale-covariant, as it computes keypoints at different scales, so we tried to emulate this for dense SIFT with another parameter. However, there is not a substantial improvement in the results in this case, so the scale is not an important factor in this dataset. For this reason, we can conclude that losing the scale-covariance property of vanilla SIFT is not a problem when using dense SIFT.

## B. Classifiers

The presented results are obtained using a k-NN classifier, which might be too simple in some tasks such as image classification. For this reason, we analyze the performance

TABLE I ACCURACY OBTAINED WITH DIFFERENT DESCRIPTORS USING K-NN

| Descriptor   | Type          | Accuracy   |
|--------------|---------------|------------|
| SIFT         | Vanilla Dense | 0.55 0.74  |
| SURF         | Vanilla       | 0.62       |
| SURF         | Dense         | 0.63       |
| DAISY        | Dense         | 0.66       |

TABLE II ACCURACY OBTAINED WITH DIFFERENT CLASSIFIERS USING DENSE SIFT

| Classifier          |   Accuracy |
|---------------------|------------|
| k-NN                |       0.74 |
| Logistic Regression |       0.79 |
| SVM                 |       0.83 |

of the system using other classifiers: a logistic regression model and a SVM. As aforementioned, the descriptor used in each case is dense SIFT.

The corresponding results, presented in Tab. II, show that SVM outperforms both k-NN and logistic regression classifiers. For this reason, we conclude that SVM is better suited for our problem, and thus it will be the one used in the rest of the experiments.

- 1) Fine-tuning SVM kernel: To further improve the results, we compute the accuracy (mean and standard deviation) for different SVM kernels. In addition to the the typical kernels (linear, poly, RBF and sigmoid), we create our own: the histogram intersection kernel, defined in Eq. 1.

$$K int ( A,B ) = m ∑ i =1 min { a i , b i } (1)$$

The results are shown in Tab. III. The worse results are obtained using the linear kernel. The reason is that, in this dataset, images of different classes share visual words (e.g. trees in both forests and open country classes), so they are not linearly separable. On the other hand, the performance using the histogram intersection kernel is acceptable, as this kernel is useful in our specific problem, where the features are histograms. However, it is recommended to use the RBF kernel , which creates non-linear combinations of the features to uplift the samples onto a higher-dimensional feature space where a linear decision boundary can be used. Indeed, we obtain the best accuracy with this kernel.

## C. Spatial pyramids

The BoVW system efficiently aggregates local features into a single global vector but it completely ignores the information about the spatial layout of the features. To tackle this, we compute the keypoints and descriptors at different pyramidal levels. Spatial pyramids work by partitioning

TABLE III ACCURACY OBTAINED WITH DIFFERENT SVM KERNELS

| Kernel                 | Accuracy   | Accuracy   |
|------------------------|------------|------------|
|                        | Mean       | Std Dev    |
| Linear                 | 0.77       | 0.03       |
| Poly                   | 0.76       | 0.03       |
| RBF                    | 0.83       | 0.02       |
| Sigmoid                | 0.81       | 0.02       |
| Histogram intersection | 0.81       | 0.02       |

Fig. 6. Top: square sub-regions for levels (a) 1 and (b) 2. Bottom: horizontal sub-regions for levels 1 (c) and 2 (d)

<!-- image -->

the image into increasingly fine sub-regions and computing histograms of the visual words inside each sub-region.

- 1) Square sub-regions: The first approach is to divide each region in 4 square sub-regions for each level (Fig. 6a,b). For example, using three levels, we first compute the descriptors for the whole image. Then, the image is divided in 4 blocks and the descriptors are computed for each of them. Each sub-block is divided into 4 blocks (16 in total) and the descriptors are computed for each of them. Finally, the descriptors computed in each level are concatenated (in this case, 1 + 4 + 16 = 21 descriptors). By computing the descriptors of the different sub-regions of the image, we can later compute the histograms of visual words of each of the regions.

2) Horizontal sub-regions: As this specific dataset is formed by a large number of landscape images, we thought it might be interesting to divide the image in horizontal subregions, as shown in Fig. 6c,d. In this case, for each level the image is divided into 3 more sub-regions. For example, using three levels, the resulting descriptor will have ( 1 + 3 + 6 = ) 10 concatenated histograms.

## D. Normalization

It is a good practice to normalize the data to avoid different scales of the feature vectors and thus improve data integrity. In this case, the histograms of visual words are normalized using L2 norm, Power norm or StandardScaler:

- · L2 norm, the modulus of the feature vector is 1:

$$x norm = x ‖ x ‖ 2 (2)$$

where

$$‖ x ‖ 2 = √ x 2 1 + x 2 2 + x 2 3 + . . . + x 2 m (3)$$

- · Power norm, the sum of all the values of the feature vector is 1:

$$x norm = x ∑ i =1 ( x i ) (4)$$

- · StandardScaler, standardize the feature vector by removing the mean and scaling to unit variance:

$$x norm = x -µ σ (5)$$

TABLE IV ACCURACY OBTAINED COMBINING DIFFERENT PYRAMID SHAPES, LEVELS AND NORMALIZATIONS

| Shape      |   Level |   None |      |   Normalization L2 Power |   StandardScaler |
|------------|---------|--------|------|--------------------------|------------------|
| Square     |       0 |   0.81 | 0.81 |                     0.81 |             0.81 |
| Square     |       1 |   0.81 | 0.82 |                     0.81 |             0.82 |
| Square     |       2 |   0.82 | 0.83 |                     0.82 |             0.83 |
| Horizontal |       0 |   0.81 | 0.81 |                     0.81 |             0.81 |
| Horizontal |       1 |   0.82 | 0.83 |                     0.83 |             0.84 |
| Horizontal |       2 |   0.82 | 0.83 |                     0.83 |             0.83 |

where µ is the mean and σ the standard deviation.

The normalization of the histograms of visual words play an important role in the spatial pyramid algorithm. If the image is divided in 4 blocks, each block has 1 / 4 of the information of the whole image. Therefore, when computing the histogram of a block, the energy (and thus the contribution) of that histogram will be 1 / 4 of the energy of the histogram of the whole image. To give the same importance to each of them, we normalize all histograms so that they contribute the same.

The results obtained using square and horizontal subregions and without and with normalization are presented in Tab. IV. As observed, for this specific dataset, if the image is divided in horizontal sub-regions , the results are better. The reason is that most of the images are landscape images with easily differentiated horizons, as aforementioned. Comparing the results for the different pyramid levels, it is observed that levels 1 and 2 outperform level 0 in all cases, which proves that features spatial information is relevant to image classification performance. On the other hand, level 2 and level 1 only differ by a small margin in most of the cases, but level 2 comes with a big extra computational cost (e.g. 21 vs 5 histograms for the square sub-regions), so level 1 is chosen as the most suitable pyramidal level to use. For higher pyramidal levels, curse of dimensionality also comes into picture as our feature space becomes sparse. Regarding normalization, we observe that the results are slightly improved in some cases, specially for the L2 norm and standardScaler. However, the improvement is not significant, which validates the consistency and integrity of our data. Even so, it is a good practice to normalize the data, and it will be useful to improve our results later, so StandardScaler normalization will be used.

## E. Clustering

To create the visual words, the multidimensional feature space is clustered using k-means, being k the codebook size (number of visual words). To further improve the results, this hyperparameter is fine-tuned, and the results are presented in Fig. 7. As observed, with a small codebook size (e.g. 32), the visual words are too general, so they are not representative enough to perform classification properly. With larger codebook sizes, the results improve up to a certain point (0.85

Fig. 7. Cross-validation accuracy for different codebook sizes

<!-- image -->

Fig. 8. Cross-validation accuracy for different PCA num\_components

<!-- image -->

for a codebook size of 512), as with more visual words each class is well represented, so it is easier for the classifier to distinct between them. The best results are obtained with a codebook size of 512 , so that will be the one used. With codebook sizes of 1024 and 2048 the results are also good, but the computational cost is much higher.

## F. Reducing dimensionality

The best results are obtained with large codebook sizes, such as 512, so we use Principal Component Analysis (PCA) to decrease the computational time. Concretely, PCA is used to reduce the feature space dimensionality projecting it to a lower dimensional space. This reduction of dimensionality is very useful in our case, as spatial pyramids increase the dimension of the vector and thus the computational time. Gridsearch is performed to fine-tune the parameter num\_components , which is used to select the number of dimensions to be kept after the dimensionality reduction.

As shown in Fig. 8, the best results are obtained with the larger num\_components (up to 0.87). This parameter defines the dimensionality of the resulting vector, and the higher its dimension, the more representative of the data and thus the better the performance of the classifier. Applying PCA slightly improves the performance, but more importantly, it speeds up the computation. For this reason, PCA with num\_components=64 will be used.

## G. Fisher vectors

Even if the BoVW approach performs well on our dataset, it finds the closest word in the vocabulary relying only on the number of local descriptors assigned to each Voronoi cell. With fisher vectors, we are not only using the mean of the local descriptors, but also including higher order statistics: the covariance. This way, the information of how far is each feature from its closest vocabulary word (and also to the other vocabulary words) is obtained.

To study its performance in our dataset, we fit an SVM classifier with the train fisher vectors (obtained from the

<!-- image -->

Fig. 9. MLP: training and validation (a) accuracy and (b) loss curves

<!-- image -->

training dataset) and we use the test fisher vectors to predict the labels of the test dataset. Fisher vectors allows training more discriminative classifiers with a lower vocabulary size. The obtained results show that encoding our feature vectors using second order information (covariances along with means) indeed benefits classification performance, as it provides similar results (0.84) reducing the computational cost.

## V. MLP

The results obtained with a classic approach such as the Bag Of Visual Words system are acceptable, but not good enough to consider the implemented image classifier robust nor reliable. For this reason, we need to use advanced techniques to improve the performance and obtain the desired results: the well-known Deep Learning (DL). As a first step towards DL, we explore the most simple architecture: Multilayer Perceptrons (MLP), in which each neuron of each layer is connected (forwards).

Using a simple MLP and a softmax layer (last layer used to predict the class of each input image), the results in terms of accuracy and loss are really bad, as shown in Fig. 9. The difference between the train and validation accuracy curves is an indicator that the model is overfitting to the training data, and thus is not able to generalize well to unseen samples (those of the test data). Moreover, the validation loss curve is unstable and not properly minimized.

To try to obtain better results, we fine-tune different parameters: learning rate, image size, number of layers (depth), layers sizes, adding normalization or regularization, and so on. Even if the performance is slightly improved in some cases, the potential of the system is limited to the fact that we are using a simple MLP for a hard image classification task, so the results will not get better. It is worth mentioning that the impact of each of these parameters (e.g. learning rate) is deeply studied in Section VII, where we design a CNN from scratch, which is more realistic in the image classification task.

## A. Deep Features, SVM and BoVW

Finally, before moving on to CNNs, we explore different variations of the MLP system in order to see if the results can be improved:

- 1) Deep Features (DF) + SVM : Extract DF from the deeper hidden layer (previous to softmax , where the features are more abstract/general) and use them to train an SVM classifier.

IMPROVEMENTS ON MLP

TABLE V ACCURACY OBTAINED USING THE BASELINE AND THE DIFFERENT

| MLP Configuration               |   Accuracy |
|---------------------------------|------------|
| Baseline                        |       0.61 |
| Optimal Learning Rate           |       0.61 |
| Different Image Sizes           |       0.64 |
| Different Layer Sizes           |       0.65 |
| Increasing Depth                |       0.66 |
| DF + SVM                        |       0.53 |
| Aggregating predictions         |       0.72 |
| DF as a dense descriptor + BoVW |       0.7  |

- 2) Aggregating predictions : Divide the input image into small patches, extract the prediction for each patch and aggregate the final prediction.
- 3) DF as a dense descriptor + BoVW Divide the input image into small patches, extract the DF from the last hidden layer for each patch, and concatenate them to create a feature vector for each image. Then, use kmeans to create a codebook and train an SVM classifier with the histograms of visual words.

The obtained results are presented in Tab. V. As observed, extracting DF and using them to train an SVM classifier is not a good alternative. Another approach is to divide each image in patches and extract deep features from each of them. In this other two cases, even if the results are slightly improved, they are not acceptable, and much worse than the ones obtained with the classic BoVW approach. For this reason, we conclude that MLP is too simple for this image classification problem.

## VI. INCEPTIONV3

The results obtained with MLP are not acceptable, so we take a step forward into deep learning to use the state of the art architecture in image-related tasks: Convolutional Neural Networks (CNNs). Since the outbreak of CNN in 2012 with AlexNet [5], multiple architectures have been presented to tackle the classification problem, obtaining increasingly better results in terms of minimizing the miss-classification error. In this paper, we fine-tune InceptionV3 [1] to adapt it to our specific dataset. This network, created by Google, is based on the idea of using Inception modules to use different sizes of channels in parallel, as there are four parallel channels in each module, which are further concatenated at the end. Specifically, each module includes factorizing convolutions with large filter size into smaller filter, factorization into asymmetric convolutions and auxiliary classifiers introduced to tackle the problem of vanishing gradient.

This model has been trained and tested with the ImageNet dataset [6], which contains around 1M images divided in 1000 classes. Therefore, it has not much to do with the dataset used in this study, and we need to adapt InceptionV3 to our specific problem.

<!-- image -->

Fig. 10. InceptionV3: training and validation (a) accuracy and (b) loss curves

<!-- image -->

<!-- image -->

Fig. 11. InceptionV3: (a) ROC curves and (b) confusion matrix

<!-- image -->

## A. Changing the network architecture

The first approach we take is to use the existing model and weights by modifying only the last layer of the architecture: the softmax layer. This is a required step to adapt the output to the number of classes that our dataset has: eight. First of all, we freeze all the layers except the last one, so that the training stage does not affect the pre-trained weights of the model.

As observed, in Fig. 10, the results are greatly improved using InceptionV3 compared to the ones obtained with a simple MLP, which proves the potential and usefulness of CNNs in image classification problems. Specifically, the difference between the training and validation loss is much lower, so there is no overfitting. Furthermore, both training and validation losses are correctly minimized.

In the confusion matrix (Fig. 11b), we observe that InceptionV3 performs really well in most of the cases, but it misclassifies a lot of forest and mountain samples as opencountry. This is also observed in the ROC curve (Fig. 11a), as the opencountry's Area Under the Curve (AUC) is lower.

## B. Unfreezing some layers

The next step is to unfreeze and retrain some layers of InceptionV3, to see if the learned weights improve the results. InceptionV3 is divided in 11 Inception blocks and a total of 311 layers, so it is not easy to select which layers to unfreeze. For this reason, we unfreeze by blocks (always starting from the end).

The results are presented in Tab. VI. As expected, the test accuracy increases as we unfreeze and retrain more blocks of the model, being the best result the one obtained with the full retrained model. However, the number of trainable parameters also increases, so the computational cost is much higher.

## ACCURACY AND PARAMETERS FOR DIFFERENT NUMBER OF UNFREEZED

TABLE VI INCEPTION BLOCKSTABLE VII ACCURACY AND PARAMETERS FOR DIFFERENT NUMBER OF REMOVED INCEPTION BLOCKS

| Unfreezed blocks   | Parameters (Total: 21.82M) Trainable   | Non-trainable   |   Accuracy |
|--------------------|----------------------------------------|-----------------|------------|
| None               | 16.3K                                  | 21.81M          |       0.89 |
| 1                  | 6.09M                                  | 15.73M          |       0.91 |
| 3                  | 12.83M                                 | 8.99M           |       0.92 |
| 5                  | 16.66M                                 | 5.16M           |       0.94 |
| 7                  | 19.64M                                 | 2.18M           |       0.93 |
| All                | 21.78M                                 | 34.43K          |       0.96 |

| Removed blocks Parameters Epochs Accuracy   | Removed blocks Parameters Epochs Accuracy   |   Removed blocks Parameters Epochs Accuracy |   Removed blocks Parameters Epochs Accuracy |
|---------------------------------------------|---------------------------------------------|---------------------------------------------|---------------------------------------------|
| None                                        | 21.82M                                      |                                          20 |                                        0.96 |
| 1                                           | 15.7                                        |                                          40 |                                        0.96 |
| 3                                           | 8.9M                                        |                                          50 |                                        0.94 |
| 5                                           | 5M                                          |                                          60 |                                        0.96 |
| 7                                           | 2.1M                                        |                                          70 |                                        0.9  |

## C. Removing blocks of layers from InceptionV3

In order to reduce the number of parameters, we decide to remove some Inception blocks and study the performance of the new (retrained) model in our specific dataset. To do so, we get the output from an specific block (e.g. block #3, so that 8 blocks are removed), add a globalaveragepooling2d layer and a final softmax layer. In each case, the full model is retrained and it takes a different number of epochs to converge.

As observed in Tab. VII, by removing 5 blocks we still obtain a really high accuracy, and we reduce the number of parameters from 21.82M to 5M, more than four times less, which makes our system lighter. This is possible because our dataset is much simpler than ImageNet, as it only has 8 different classes, so the model needs a much smaller number of parameters to learn to predict them.

## D. Tiny dataset

Once the architecture of our model has been lightened, we train it with a smaller dataset (50 samples per class, making a total of 400 samples) to study its performance. This is presented in Fig. 12. As expected, the model is not able to learn that good (nor fast) using the tiny dataset, as it needs more samples to correctly set the weights of each layer, and the resulting accuracy is lower.

To improve the learning of our new model with the tiny dataset, we introduce and evaluate the usage of data augmentation. To do so, we use different augmentations individually and combined, to see if adding more variability to our training data improves the performance of the new model.

<!-- image -->

Fig. 12. New model trained with tiny dataset: training and validation (a) accuracy and (b) loss curves

<!-- image -->

TABLE VIII ACCURACY FOR DIFFERENT DATA AUGMENTATIONS IN THE TINY DATASET

| Data augmentation    |      |   Value Accuracy |
|----------------------|------|------------------|
| None                 | -    |             0.9  |
| Horizontal Flip (HF) | True |             0.93 |
| Zoom (Z)             | 20%  |             0.92 |
| Rotate (R)           | 10º  |             0.92 |
| Shear (S)            | 20%  |             0.92 |
| Width Shift (WS)     | 20%  |             0.92 |
| Height Shift (HS)    | 20%  |             0.92 |

The results, shown in Tab. VIII, show that each one of the augmentation methods is helping our model, so data augmentation is very useful to contribute to the variability of the training data and thus help with the learning. However, combining the augmentations the results are not further improved, as they may distort the images too much. Therefore, horizontal flip is enough for this problem.

## E. Random search to tune the hyperparameters

The last step is to refine the model hyperparameters that optimizes the validation accuracy results using the following options:

- · Optimizer: SGD, RMSprop, Adam, Adadelta, Adagrad
- · Learning Rate: 0.001, 0.01, 0.1, 0.2
- · Momentum: 0.6, 0.8, 0.9
- · Activation function: elu, relu, tanh

Considering the size of our network, we cannot do an exhaustive gridsearch, as it is not feasible in terms of computational time, so we use the random search implementation from keras tuner.

<!-- image -->

Fig. 13. Final model trained with tiny dataset: training and validation (a) accuracy and (b) loss curves. Red line: original InceptionV3 results

<!-- image -->

<!-- image -->

Fig. 14. Final model trained with tiny dataset: (a) ROC curves and (b) confusion matrix

<!-- image -->

Fig. 15. Baseline architecture

<!-- image -->

The best results are obtained with the SGD optimizer, learning rate of 0.001, momentum of 0.9 and relu activation function. The resulting accuracy and loss curves are presented in Fig. 13. As expected, our model needs more epochs to converge compared to the original InceptionV3, as now we are using a much smaller dataset. However, the resulting accuracy is higher, and the loss is correctly minimized, which proves that the network has been correctly fine-tuned for our specific case.

The ROC curve and confusion matrix are also presented in Fig. 14. As observed, the new model has improved the performance in the classes that were more difficult for the original InceptionV3, as it is not misclassifying forest and mountain samples as opencountry anymore. However, the recall of the opencountry class has decreased, so this model could be further improved.

## VII. DESIGNING OUR OWN CNN

To better fit the model to the problem, we design a CNN from scratch. The baseline of our network is formed by two blocks of a 2D convolutional layer and a 2D max pooling, followed by a dense output layer with a 'softmax' activation function. The model is represented in Fig. 15.

The performance of this baseline model can be seen in terms of accuracy and loss in Fig. 16.

We obtain an accuracy of 0.78, which is already more than what we get with the MLP. However, the accuracy curve shows overfitting and the loss curve is unstable and starts diverging for the validation set.

The parameters used on the convolutional layers are the default ones by Keras examples: Kernel size of 5x5, 32

<!-- image -->

Fig. 16. Baseline (a) accuracy and (b) loss curves

<!-- image -->

TABLE IX TUNING THE KERNEL SIZE

| Kernel Size   |   Accuracy |   Loss |   # Parameters |   Ratio |
|---------------|------------|--------|----------------|---------|
| 1x1           |       0.62 |   1.17 |          17576 |    3.55 |
| 3x3           |       0.77 |   0.53 |          19368 |    3.97 |
| 5x5           |       0.78 |   1.2  |          34472 |    2.2  |
| 7x7           |       0.78 |   0.86 |          57256 |    1.37 |

filters, Relu activation function and Glorot Normal weight initialization.

## A. Kernel Size

In order to improve our system, we tune the different parameters of the convolutional layers to find out the limits of this baseline. The first parameter to be tuned is the kernel size, obtaining the results in Tab. IX. The best accuracy is obtained with the kernel sizes of 5x5 and 7x7. However, for our CNN we introduce another metric to take into account: the accuracy-parameter ratio, that can be calculated with Eq. 6. Taking into account this ratio, the best compromise between accuracy, loss and ratio is obtained with a 3x3 kernel , and therefore we will use this one for the following tests. This is not really a surprise, as since VGG [7] introduced the usage of this size of kernel, it has somehow become an standard. For example, two layers of a 3x3 kernel produce better results than one with a 5x5 kernel size.

$$ratio = accuracy ∗ 10 5 number of parameters (6)$$

## B. Number of filters

Changing the kernel size we improve our ratio, but slightly worsen the accuracy. Hence more changes are needed. We tune the number of filters used on both convolutional layers, obtaining the results shown in Tab. X. Again the parameter that gives a better accuracy, 64 filters, reduces considerably the ratio.

Instead of using the same number of filters for both layers, we can combine two different number of filters. The best combination we found is using 64 filters for the first layer and 32 for the second, which performs with an accuracy of 0.78, 29480 parameters and a ratio of 2.64. This results improve the ratio from the baseline while maintaining the accuracy, thus we use this configuration from now on.

TABLE X TUNING THE NUMBER OF FILTERSTABLE XI TUNING THE ACTIVATION FUNCTIONS

|   # of Filters |   Accuracy |   Loss |   # Parameters |   Ratio |
|----------------|------------|--------|----------------|---------|
|              8 |       0.71 |   0.63 |           3120 |   22.69 |
|             16 |       0.76 |   0.58 |           7384 |   10.28 |
|             32 |       0.77 |   0.53 |          19368 |    3.97 |
|             64 |       0.79 |   0.35 |          57160 |    1.38 |
|            128 |       0.76 |   0.51 |         188040 |    0.4  |

TABLE XII

| Activation   |   Accuracy |   Loss |   # Parameters |   Ratio |
|--------------|------------|--------|----------------|---------|
| Relu         |       0.78 |   0.56 |          29480 |    2.64 |
| Elu          |       0.76 |   0.85 |          29480 |    2.58 |
| Tanh         |       0.74 |   0.74 |          29480 |    2.5  |

TUNING THE WEIGHT INITIALIZATION

| Initialization   |   Accuracy |   Loss |   # Parameters |   Ratio |
|------------------|------------|--------|----------------|---------|
| Glorot Uniform   |       0.78 |   0.56 |          29480 |    2.64 |
| Glorot Normal    |       0.78 |   0.52 |          29480 |    2.64 |
| He Normal        |       0.76 |   0.82 |          29480 |    2.57 |
| Random Normal    |       0.77 |   0.8  |          29480 |    2.61 |
| Zeros            |       0.14 |   2.1  |          29480 |    0.5  |

## C. Activation functions

The next parameter to be tuned is the activation function. In this case we can observe that the default activation function, ReLU , gives the best results as seen in Tab.XI. Therefore, changing the activation will not improve our system.

## D. Weight initialization

Finally, we tune the weight initialization. Glorot, He and Random initialization are compared in Tab. XII. We add as well the pitfall: all zero initialization. We can empirically validate that what we should not initialize the weights to 0, as accuracy drops to 0.14. The reason is that by doing so, every neuron in the network computes the same output, the same gradients during backpropagation and undergo the exact same parameter updates. In other words, there is no source of asymmetry between neurons if their weights are initialized to be the same. Comparing the other initializations, numerically Glorot normal and uniform perform almost identically. To visualize the improvement in performance we need to check the accuracy and loss curves (Fig. 17). We can observe how with Glorot normal initialization, the loss curve (Fig. 17d) converges in a stable and smooth way. For this reason, Glorot normal initialization will be the one used.

Still our system did not improve its performance significantly, and we can state that continue tuning the baseline hyper-parameters is not the path to follow.

## E. Adding depth

Another option is to increase the depth of our architecture. The first approach is to keep adding blocks of 2D convo-

Fig. 17. Glorot (a) uniform and (b) normal accuracy curves; Glorot (c) uniform and (d) normal loss curves

<!-- image -->

TABLE XIII

## ADDING DEPTH TO THE CNN

TABLE XIV

| Architecture   |   Accuracy |   Loss |   # Parameters |   Ratio |
|----------------|------------|--------|----------------|---------|
| Baseline       |       0.77 |   0.53 |          19368 |    3.97 |
| Three layers   |       0.8  |   0.75 |          20424 |    3.94 |
| Four layers    |       0.81 |   0.8  |          29672 |    2.48 |
| Five layers    |       0.77 |   0.67 |          38152 |    2.03 |

## TUNING THE OPTIMIZERS

TABLE XV

| Optimizer   |   Accuracy |   Loss |   # Parameters |   Ratio |
|-------------|------------|--------|----------------|---------|
| RMS prop    |       0.74 |   0.52 |          66344 |    1.12 |
| Adam        |       0.82 |   0.68 |          66344 |    1.25 |
| SGD         |       0.81 |   0.75 |          66344 |    1.22 |

lutional + 2d max pooling layers to our architecture. We use the hyperparameters obtained in the baseline tuning: 3x3 kernel size, Relu activation and Glorot normal initialization. However, in terms of number of filters we go back to using 32 of them for all the layers. In Tab. XIII we can observe how at first adding layers increases the accuracy up to 0.81, but when adding a fifth one the performance does not improve but even get worse.

Again adding depth randomly works but only up to a certain point. Additionally we only added one kind of layers, when there are a lot of possibilities to test. Using the fourth layer architecture, we add a dropout and a batch-norm layer, even though there is a consensus in not using them together. The final architecture is shown in Fig. 18, and with it we get an accuracy of 0.82 with 66344 parameters and a ratio of 1.25.

## F. Optimizer and Learning rate

So far we ignored the hyper-parameters related to the optimizers but is another factor to take into account. Comparing the values obtained on Tab. XIV we can observe that Adam provides the best results.

Strongly related to the optimizer, we have the learning rate, and it is usually hard to establish the best value. Numerically,

Fig. 18. Deep architecture

<!-- image -->

TUNING THE LEARNING RATE

|   Learning Rate |   Accuracy |   Loss |   # Parameters |   Ratio |
|-----------------|------------|--------|----------------|---------|
|          0.1    |       0.79 |   0.85 |          66344 |    1.19 |
|          0.01   |       0.8  |   0.72 |          66344 |    1.21 |
|          0.001  |       0.82 |   0.64 |          66344 |    1.25 |
|          0.0001 |       0.83 |   0.55 |          66344 |    1.25 |

we can observe in Tab. XV that the smaller the learning rate the better results we obtain. But it is graphically that we can extract more information on how the system behaves. We can observe in Fig. 19 that the loss values get lower as the learning rate is decreased. In addition, for a learning rate of 1e-4, we observe that the accuracy curve (Fig. 19g) has no overfitting.

## G. Input size

It is not just the architecture of the system that affects the performance. How we preprocess the input data can play an important role too. In this case, we validate it by changing the size of the input images. We can observe that the performance changes significantly between the different sizes(Tab. XVI), and it is with an input size of 64x64 that we obtain the best accuracy so far: 0.84.

Fig. 19. Left column shows the accuracy curve and the right one the loss curve for learning rates of (a,b) 1e-1, (c,d) 1e-2, (e,f) 1e-3 and (g,h) 1e-4

<!-- image -->

TABLE XVI TUNING THE INPUT SIZE

|   Input Size |   Accuracy |   Loss |   # Parameters |   Ratio |
|--------------|------------|--------|----------------|---------|
|           16 |       0.74 |   0.78 |          66344 |    1.12 |
|           32 |       0.83 |   0.55 |          66344 |    1.25 |
|           64 |       0.84 |   0.56 |          66344 |    1.26 |
|          128 |       0.79 |   0.63 |          66344 |    1.19 |

## H. Grad-CAM

It might feel that the deeper we go, the system becomes a darker box and it is harder to understand how every layer contributes our system. Fortunately, we can use techniques like Grad-CAM [8] to visualize the regions of the input data that are more relevant for predicting an specific concept. In Fig. 20a the activation for the forest class highlights the trees of the image. For the tall building class, in Fig. 20b, it is clear that the highlighted object is the sky-scrapper. On the other hand, for the mountain example (Fig. 20c) is the silhouette of the peak that allows the system to predict it correctly. Finally, we have an example of the open country class in

Fig. 20. Examples of activation maps for the (a) forest, (b) tall-building, (c) mountain and (d) open country classes

<!-- image -->

Fig. 21. Problems caused by poor initialization in neural networks. Network layers (a to f): 1 (shallow) to 6 (deeper). For deeper layers, activations start becoming zero and the gradients start collapsing to zero as well in the weights are not initialized properly

<!-- image -->

Fig 20d, which is, as we have seen before, the most difficult class to classify and there is not a distinctive activation to relate it to.

## VIII. REVISITING WEIGHT INITIALIZATION

We want to answer the question of why a good initialization matters in neural networks? As Neural Networks involve a lot of matrix multiplications, the mean and variance of activations can quickly shoot off to very high values or drop down to zero as seen in Fig. 21. This will cause the local gradients of our layers to become NaN or zero and hence prevent our network from learning anything as the value of gradients depend on the forward activations as seen in Fig. 22. A common strategy to avoid this is to initialize the weights of your network using the latest techniques. For example if you're using ReLU activation after a layer, you must initialize your weights with Kaiming He initialization and set the biases to zero. This was introduced in the 2014 ImageNet winning paper [9] from Microsoft. This ensures the mean and standard deviation of activations of all layers stay close to 0 and 1 respectively.

As you can see in Fig. 23, the gradients in both sigmoid and tanh are non-zero only inside a certain range between [5, 5]. Also notice that when using sigmoid, the local gradient

Fig. 22. Upstream gradients are multiplied by local gradients to get the downstream gradients during backprop

<!-- image -->

<!-- image -->

<!-- image -->

Fig. 23. Comparing the local gradient behavior of some common activation functions. (a)Sigmoid (b)Tanh (c)ReLU (d)Softplus

<!-- image -->

Fig. 24. Stacking blocks that make our TinyNet, followed by adaptive average pooling

<!-- image -->

achieves a maximum value of 0.25, thus every time gradient passes through a sigmoid layer, it gets diminished by at least 75 percent.

## IX. BRINGING EVERYTHING TOGETHER: TINYNET

## A. Main Architecture

We will be using the insight gained from all our previous experiments to train a 4-"block" CNN where each block may be composed of convolutions, activations, batchnorm, and residual connections. We will be using an input image size of 64x64, 3x3 kernels for each layer, stride 2 with padding 1 for downsampling (we will not be using max-pooling layers but only adaptive average pooling in the end layer). The outline for our TinyNet can be seen in Fig. 24. The size of layers used are [32 , 64 , 128 , 256] .

We track the performance of our network by keeping a track of the means and the variances of activations as our network trains. You can compare the network's performance for each of the block configuration we experiment with in Fig. 25 and compare their accuracies in Table XVII.

However these values are just aggregates of the layer parameters, so they don't give us the full picture about how all the parameters are behaving. Rather than look at a single

<!-- image -->

Fig. 25. Layer activations: (left) mean and (right) standard deviation. Initializations: (a,b) Glorot uniform, (c,d) Kaiming He, (e,f) He + Leaky ReLU, and (g,h) He + LeakyReLU + BatchNorm

<!-- image -->

number we'd like to look at the distribution. To do that we can look at how the histogram of the parameters changes over time as shown in Fig. 26. The biggest concern is the amount of mass at the bottom of the histogram (at 0) in the original network. This is not good. In the last layer nearly 90 percent of the activations are actually 0. If you were training your model like this, it could appear like it was learning something, but you could be leaving a lot of performance on the table by wasting 90 percent of your activations. But, by using proper initiliazation and training techniques this can be fixed as we show. Notice in Fig. 26b, it's using the full richness of the possible activations and there's not crashing of values.

Now that we know how to train our networks properly, let's see how it performs for different depth and width. We show the results in Table XVIII.

TABLE XVII TRAIN AND TEST ACCURACY FOR DIFFERENT "BLOCKS" OF OUR MODEL

| Model                       |   Train | Test          |
|-----------------------------|---------|---------------|
| Keras Default (Glorot)      |   0.842 | 0.813         |
| Kaiming He init             |   0.895 | 0.874 (+0.06) |
| He + Leaky ReLU             |   0.887 | 0.883 (+0.07) |
| He + Leaky ReLU + BatchNorm |   0.912 | 0.901 (+0.09) |

Fig. 26. Histogram of Activations for each layer. From first layer(top) to last layer(bottom) (a) Keras Default (Glorot). (b) Kaiming He

<!-- image -->

## B. Adding Residual connections

We added residual connections[9] (Fig. 27) in our network but they didn't improve our accuracy at all, even though the number of parameters increased by 3 times. This was surprising for us, but the explanation is clear. As shown in Fig. 28, this time the bottleneck was not the model, but the amount of data that we had. With just 1800 training samples, it is difficult to train a good model from scratch. Even after using a lot of data augmentation, it wasn't enough to train a deeper model.

Now that it was certain that making the model bigger won't be of much use, we shifted out attention to making the model smaller and more efficient.

## C. Adding Depthwise Convolutions

We replace normal convolutions in our model with depthwise convolutions, adapting this idea from MobileNets [10].

BEST CONFIGURATION (CONV BLOCK WITH KAIMING INITIALIZATION + LEAKY RELU + BATCHNORM) USED FOR THESE EXPERIMENTS.

TABLE XVIII

|   Layers | Filters               | # Params   |   Train |   Test |   Ratio |
|----------|-----------------------|------------|---------|--------|---------|
|        4 | [32 , 64 , 128 , 256] | 390,952    |   0.912 |  0.901 |    0.23 |
|        4 | [16 , 32 , 64 , 128]  | 98,712     |   0.903 |  0.884 |    0.91 |
|        3 | [32 , 64 , 128]       | 94,504     |   0.893 |  0.873 |    0.95 |
|        3 | [16 , 32 , 64]        | 24,216     |   0.864 |  0.861 |    3.58 |

Fig. 27. Residual Connection

<!-- image -->

## BIG DATA & DEEP LEARNING

Fig. 28. Increased performance with increasing data for deep learning

<!-- image -->

The MobileNet model is based on depthwise separable convolutions which is a form of factorized convolutions which factorize a standard convolution into a depthwise convolution and a 1×1 convolution called a pointwise convolution. This factorization has the effect of drastically reducing computation and model size. By expressing convolution as a two step process of filtering and combining, we get a reduction in computation per layer shown in Eq. 7:

$$reduction = 1 N + 1 D 2 k (7)$$

where N is the number of kernels in the filter, and D k is the size of the kernel. For kernel size of 3x3, we can get approx. 9 times reduction in model size. A visual explanation of the difference is given in Fig. 29.

As we can see in the last row of the Table XIX, we are able to get an accuracy of 0.825 only with as low as 4,000 parameters.

## D. Optimizers

We have used a few state-of-the-art tricks to automatically find the best learning rate and momentum for our Adam Optimizer. This includes the One Cycle Policy, [11] and Learning Rate Finder[12] proposed by Leslie N. Smith that allowed us to train our networks at much higher learning rates, and thus they converge in lower number of epochs. Each epoch took only 3s to run and each of our models converged in less than 25 epochs, making the whole experiment around 1.5 mins. This was a big factor in our project as it allowed us to rapidly prototype and experiment different configurations and ideas.

## X. CONCLUSIONS

In this paper, we have explored some traditional and Deep Learning techniques that can be used in a classification system, and we have seen the advantages and disadvantages

TABLE XIX DEPTHWISE CONVOLUTIONS REPLACING NORMAL CONVOLUTIONS IN THE MODELS PRESENTED IN TABLE XVII

| Type           |   Layers | Filters               | # Params   |   Train |   Test |   Ratio |
|----------------|----------|-----------------------|------------|---------|--------|---------|
| Normal Conv    |        4 | [32 , 64 , 128 , 256] | 390,952    |   0.912 |  0.901 |    0.23 |
| Depthwise Conv |        4 | [32 , 64 , 128 , 256] | 48,617     |   0.894 |  0.874 |    1.8  |
| Normal Conv    |        4 | [16 , 32 , 64 , 128]  | 98,712     |   0.903 |  0.884 |    0.91 |
| Depthwise Conv |        4 | [16 , 32 , 64 , 128]  | 13,577     |   0.873 |  0.863 |    6.4  |
| Normal Conv    |        3 | [16 , 32 , 64]        | 24,216     |   0.864 |  0.861 |    3.58 |
| Depthwise Conv |        3 | [16 , 32 , 64]        | 3,913      |   0.836 |  0.825 |   21.08 |

Fig. 29. (a) Normal and (b) Depthwise convolution

<!-- image -->

of using some models compared to others. From the results obtained, we can draw the following conclusions.

First of all, classic approaches (e.g. BoVW) can provide good results, but they are limited and would not make a reliable nor robust system for a real world application. However, it still performs better than a simple Deep Learning technique like the Multi-Layer Perceptron. MLPs are too simple for our image classification problem, and even when optimizing their hyper-parameters the performance is poor. Using the MLP system to extract deep features to use afterwards on SVM, or as descriptors for the BoVW system does not provide us with better results neither.

Fine-tuning on a pre-trained network (e.g. InceptionV3) gives the best results in terms of accuracy when unfreezing and retraining the weights, 96 percent. But if we take into account the number of parameters, it is not an efficient system. Indeed, we can remove layers and reduce by five times the amount of parameters without losing accuracy, and still be an overkill for our specific dataset.

When training a model from scratch, a good weight initialization is important for making sure our models train properly. We need a good amount of data for training deep neural networks, and with the dataset that we had (1,800 samples) we could not improve our accuracy above a threshold (90 percent), because the data wasn't enough to learn features that were representative enough of the dataset. Still our model is better fitted to the specific problem and dataset: while the pruned InceptionV3 had 5M parameters,

in our model we have around 4K parameters.

## REFERENCES

- [1] C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens, and Z. Wojna. Rethinking the inception architecture for computer vision. In 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , pages 2818-2826, 2016.
- [2] David Lowe. Distinctive image features from scale-invariant keypoints. International Journal of Computer Vision , 60:91-, 11 2004.
- [3] Herbert Bay, Andreas Ess, Tinne Tuytelaars, and Luc Van Gool. Speeded-up robust features (surf). Computer Vision and Image Understanding , 110(3):346-359, 2008. Similarity Matching in Computer Vision and Multimedia.
- [4] E. Tola, V. Lepetit, and P. Fua. Daisy: An efficient dense descriptor applied to wide-baseline stereo. IEEE Transactions on Pattern Analysis and Machine Intelligence , 32(5):815-830, 2010.
- [5] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton. Imagenet classification with deep convolutional neural networks. In Proceedings of the 25th International Conference on Neural Information Processing Systems - Volume 1 , NIPS'12, page 1097-1105, Red Hook, NY, USA, 2012. Curran Associates Inc.
- [6] J. Deng, W. Dong, R. Socher, L. Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE Conference on Computer Vision and Pattern Recognition , pages 248-255, 2009.
- [7] Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. In International Conference on Learning Representations , 2015.
- [8] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra. Grad-cam: Visual explanations from deep networks via gradient-based localization. In 2017 IEEE International Conference on Computer Vision (ICCV) , pages 618-626, 2017.
- [9] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Delving deep into rectifiers: Surpassing human-level performance on imagenet classification. CoRR , abs/1502.01852, 2015.
- [10] Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, and Hartwig Adam. Mobilenets: Efficient convolutional neural networks for mobile vision applications. CoRR , abs/1704.04861, 2017.
- [11] Leslie N. Smith. No more pesky learning rate guessing games. CoRR , abs/1506.01186, 2015.
- [12] Leslie N. Smith. A disciplined approach to neural network hyperparameters: Part 1 - learning rate, batch size, momentum, and weight decay. CoRR , abs/1803.09820, 2018.