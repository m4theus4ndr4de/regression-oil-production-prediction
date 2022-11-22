
<img src="image/extraction.jpg" alt="logo" style="zoom:100%;" />

<h1>Oil Production Prediction</h1>

<p align="justify">This is a fictional project for studying purposes. The business context and the insights are not real.

<h2>1. Description of the Business Problem</h2>

<p align="justify">Production prediction is one of the core problems in a company. The provided dataset is a set of nearby wells located in the United States and their 12 months cumulative production. The company needs a production prediction model to serve as one of the tools to support the company decisions. So, the company data scientist needs to build a model from scratch to predict production and show the manager that the model can perform well on unseen data.</p>

<h3>The tools that were created:</h3>

<p align="justify"><b>Machine Learning Regression Model: </b>Using the dataset provided by the company. A machine learning regression model was created to be used for future predictions.</p> The notebook used to create the model is available <a href="https://github.com/m4theus4ndr4de/regression-oil-production-prediction/blob/main/notebook/production_prediction.ipynb" target="_blank">here</a>.</p>

<p align="justify"><b>Streamlit App for Production Prediction: </b>The model is available on the Streamlit Cloud and can be used through the Streamlit App created. The App is available <a href="https://m4theus4ndr4de-regression-oil-production--prediction-app-lhyr2y.streamlit.app/" target="_blank">here</a>.</p>

<h2>2. Dataset Attributes</h2>

<table style="width:100%">
<tr><th>Attribute</th><th>Description</th></tr>
<tr><td>treatment company</td><td>The treatment company who provides treatment service.</td></tr>
<tr><td>azimuth</td><td>Well drilling direction.</td></tr>
<tr><td>md (ft)</td><td></td></tr>
<tr><td>tvd (ft)</td><td>True vertical depth.</td></tr>
<tr><td>date on production</td><td>First production date.</td></tr>
<tr><td>operator</td><td>The well operator who performs drilling service.</td></tr>
<tr><td>footage lateral length</td><td>Horizontal well section.</td></tr>
<tr><td>well spacing</td><td>Distance to the closest nearby well.</td></tr>
<tr><td>porpoise deviation</td><td>How much max (in ft.) a well deviated from its horizontal.</td></tr>
<tr><td>porpoise count</td><td>How many times the deviations (porpoises) occurred.</td></tr>
<tr><td>shale footage</td><td>How much shale (in ft) encountered in a horizontal well.</td></tr>
<tr><td>acoustic impedance</td><td>The impedance of a reservoir rock (ft/s * g/cc).</td></tr>
<tr><td>log permeability</td><td>The property of rocks that is an indication of the ability for fluids (gas or liquid) to flow through rocks.</td></tr>
<tr><td>porosity</td><td>The percentage of void space in a rock.</td></tr>
<tr><td>poisson ratio</td><td>Measures the ratio of lateral strain to axial strain at linearly elastic region.</td></tr>
<tr><td>water saturation</td><td>The ratio of water volume to pore volume.</td></tr>
<tr><td>toc</td><td>Total Organic Carbon, indicates the organic richness (hydrocarbon generative potential) of a reservoir rock.</td></tr>
<tr><td>vcl</td><td>The amount of clay minerals in a reservoir rock.</td></tr>
<tr><td>p-velocity</td><td>The velocity of P-waves (compressional waves) through a reservoir rock (ft/s).</td></tr>
<tr><td>s-velocity</td><td>The velocity of S-waves (shear waves) through a reservoir rock (ft/s).</td></tr>
<tr><td>youngs modulus</td><td>The ratio of the applied stress to the fractional extension (or shortening) of the reservoir rock parallel to the tension (or compression) (giga pascals).</td></tr>
<tr><td>isip</td><td>When the pumps are quickly stopped, and the fluids stop moving, these friction pressures disappear and the resulting pressure is called the instantaneous shut-in pressure, ISIP.</td></tr>
<tr><td>breakdown pressure</td><td>The pressure at which a hydraulic fracture is created/initiated/induced.</td></tr>
<tr><td>pump rate</td><td>The volume of liquid that travels through the pump in a given time.</td></tr>
<tr><td>total number of stages</td><td>Total stages used to fracture the horizontal section of the well.</td></tr>
<tr><td>proppant volume</td><td>The amount of proppant in pounds used in the completion of a well (lbs).</td></tr>
<tr><td>proppant fluid ratio</td><td>The ratio of proppant volume/fluid volume (lbs/gallon).</td></tr>
<tr><td>production</td><td>The 12 months cumulative gas production (mmcf).</td></tr>
</table>

<h2>3. Solution Strategy</h2>

<ol>
<li>Understand the Business problem.</li>
<li>Clean the dataset removing outliers, NA values and unnecessary features.</li>
<li>Explore the data to create hypothesis, think about a few insights and validate them.</li>
<li>Prepare the data to be used by the modeling algorithms encoding variables, splitting train and test dataset and other necessary operations.</li>
<li>Create the models using machine learning algorithms.</li>
<li>Evaluate the created models to find the one that best fits to the problem.</li>
<li>Tune the model to achieve a better performance.</li>
<li>Deploy the model in production so that it is available to other people.</li>
<li>Find possible improvements to be explored in the future.</li>
</ol>

<h2>4. The Insights</h2>

<p><b>I1:</b> Wells with a greater number of stages produce more,</p>
<p><b>True:</b> This relationship doesn't apply for all values of total number of stages, but it tends to be true.</p>
<p><b>I2:</b> Wells that started producing longer ago produce less.</p>
<p><b>True:</b> Productions from newer wells are better.</p>
<p><b>I3:</b> Wells that are farther from the others produce more.</p>
<p><b>False:</b> The production doesn't increase according to the distance from other wells.</p>
<p><b>I4:</b> Wells in which more proppant were used produce more.</p>
<p><b>True:</b> More proppant indicates a greater production.</p>
<p><b>I5:</b> Wells in which the rocks have higher values of porosity produce more.</p>
<p><b>False:</b> More porosity does not mean more production.</p>

<h2>5. Machine Learning Modeling</h2>

<p align="justify">The final result of this project is a regression model. Therefore, some machine learning models were created. So, 7 models were created, Linear Regression, Lasso, SVM, Random Forest, XGBoost, LightGBM and CatBoost.

Boruta (feature selection algorithm) was used to select features for the model and 11 features were selected to the final model. The models were evaluated considering three metrics, Mean Absolute Error (MAE), Mean Absolute Percentage Error (MAPE) and Root Mean Squared Error (RMSE). The initial models performances are in the table below.</p>

<p align="justify"></p>

<table style="width:100%">
<tr><th>Model Name</th><th>MAE</th><th>MAPE</th><th>RMSE</th></tr>
<tr><td>CatBoost</td><td>502.93</td><td>0.2817</td><td>781.34</td></tr>
<tr><td>LightGBM</td><td>522.03</td><td>0.2936</td><td>806.55</td></tr>
<tr><td>XGBoost</td><td>535.10</td><td>0.3094</td><td>813.48</td></tr>
<tr><td>Random Forest</td><td>564.38</td><td>0.3281</td><td>852.23</td></tr>
<tr><td>SVM</td><td>648.01</td><td>0.4468</td><td>931.77</td></tr>
<tr><td>Linear Regression</td><td>679.33</td><td></td><td>1012.51</td></tr>
<tr><td>Lasso</td><td>1018.08</td><td>0.4259</td><td>1396.98</td></tr>
</table>

<h2>6. Final Model</h2>

<p align="justify">To decide which would be the final model, a cross-validation was carried out to evaluate the performance of the algorithms in a more robust way. These metrics are represented in the table below.</p>

<table style="width:100%">
<tr><th>Model Name</th><th>MAE</th><th>MAPE</th><th>RMSE</th></tr>
<tr><td>Linear Regression</td><td>687.8 +/- 49.40</td><td>0.49 +/- 0.04</td><td>974.12 +/- 90.88</td></tr>
<tr><td>Lasso</td><td>1023.65 +/- 61.45</td><td>0.89 +/- 0.06</td><td>1348.19 +/- 96.97</td></tr>
<tr><td>SVM</td><td>651.62 +/- 28.27</td><td>0.51 +/- 0.06</td><td>897.34 +/- 60.87</td></tr>
<tr><td>Random Forest</td><td>521.82 +/- 26.99</td><td>0.36 +/- 0.02</td><td>768.7 +/- 74.63</td></tr>
<tr><td>XGBoost</td><td>526.78 +/- 14.36</td><td>0.35 +/- 0.02</td><td>773.11 +/- 52.73</td></tr>
<tr><td>LightGBM</td><td>525.71 +/- 31.97</td><td>0.34 +/- 0.02</td><td>767.4 +/- 58.25</td></tr>
<tr><td>CatBoost</td><td>490.18 +/- 16.5</td><td>0.32 +/- 0.02</td><td>724.79 +/- 54.17</td></tr>
</table>

<p align="justify">As the table presents, the Catboost model was the best one and was chosen to be deployed. After choosing which would be the final model, a random search hyperparameter optimization algorithm was used to improve the performance of the model. The final model evaluation metrics are in the table below.</p>

<table style="width:100%">
<tr><th>Model Name</th><th>MAE</th><th>MAPE</th><th>RMSE</th></tr>
<tr><td>CatBoost Tuned</td><td>485.66 +/- 23.01</td><td>0.32 +/- 0.02</td><td>714.4 +/- 64.6</td></tr>
</table>

<h2>7. Conclusion</h2>

<p align="justify">Although the dataset has many features, it is small and has a significant amount of missing values. The model presented a larger error than expected, this problem could be circumvented with a larger amount of data. Using the app, other people can easily make predictions just setting the values and pressing the prediction button.</p>

<h2>8. Future Work</h2>

<ul>
<li>Find a better way to replace missing values.</li>
<li>Find the best way of dealing with the outliers.</li>
<li>Search for models that could perform better with this small dataset.</li>
<li>Improve the <a href="https://m4theus4ndr4de-regression-oil-production--prediction-app-lhyr2y.streamlit.app/" target="_blank">Streamlit app</a> adding more functions.</li>
</ul>
