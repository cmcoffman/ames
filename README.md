# Predicting Real Estate Prices
## Background
Real estate is a large sector of the economy and home purchases and sales are probably the largest and most complex financial transaction someone will make in their lifetime. As such buyers and the lenders who finance the purchase have many layers or inspection and assessment to attempt precisely determine if the listed price for a home is reflective of it's value. There is also a significant interest in home values for the proper assessment of property taxes. In Iowa real estate is [assessed](https://tax.iowa.gov/iowa-property-tax-overview) on odd-numbered years. The assessment may or may not involve an on-site visit by an assessor and is done in a [defined, systematic manner](http://publications.iowa.gov/32396/).

## Problem statement
In this project I am tasked with examining at dataset homes sold from 2006-2010 in Ames, Iowa and building a linear model to predict sale price. Predictions will also be submitted to a [Kaggle competition](https://www.kaggle.com/competitions/221-ames-competition/). We are instructed to not use any techniques outside of what we've covered in class. Predictions on Kaggle are scored on root mean square error (RMSE).

## Data Dictionary
### Features included in the dataset
| column name     | data type  | dtype   | data description                                                       |
|-----------------|------------|---------|------------------------------------------------------------------------|
| id              | Discrete   | int64   | Observation number                                                     |
| pid             | Nominal    | int64   | Parcel identification number                                           |
| ms_subclass     | Nominal    | int64   | Identifies the type of dwelling involved in the sale.                  |
| ms_zoning       | Nominal    | object  | Identifies the general zoning classification of the sale.              |
| lot_frontage    | Continuous | float64 | Linear feet of street connected to property                            |
| lot_area        | Continuous | int64   | Lot size in square feet                                                |
| street          | Nominal    | object  | Type of road access to property                                        |
| alley           | Nominal    | object  | Type of alley access to property                                       |
| lot_shape       | Ordinal    | object  | General shape of property                                              |
| land_contour    | Nominal    | object  | Flatness of the property                                               |
| utilities       | Ordinal    | object  | Type of utilities available                                            |
| lot_config      | Nominal    | object  | Lot configuration                                                      |
| land_slope      | Ordinal    | object  | Slope of property                                                      |
| neighborhood    | Nominal    | object  | Physical locations within Ames city limits (map available)             |
| condition_1     | Nominal    | object  | Proximity to various conditions                                        |
| condition_2     | Nominal    | object  | Proximity to various conditions (if more than one is present)          |
| bldg_type       | Nominal    | object  | Type of dwelling                                                       |
| house_style     | Nominal    | object  | Style of dwelling                                                      |
| overall_qual    | Ordinal    | int64   | Rates the overall material and finish of the house                     |
| overall_cond    | Ordinal    | int64   | Rates the overall condition of the house                               |
| year_built      | Discrete   | int64   | Original construction date                                             |
| year_remod_add  | Discrete   | int64   | Remodel date (same as construction date if no remodeling or additions) |
| roof_style      | Nominal    | object  | Type of roof                                                           |
| roof_matl       | Nominal    | object  | Roof material                                                          |
| exterior_1st    | Nominal    | object  | Exterior covering on house                                             |
| exterior_2nd    | Nominal    | object  | Exterior covering on house (if more than one material)                 |
| mas_vnr_type    | Nominal    | object  | Masonry veneer type                                                    |
| mas_vnr_area    | Continuous | float64 | Masonry veneer area in square feet                                     |
| exter_qual      | Ordinal    | object  | Evaluates the quality of the material on the exterior                  |
| exter_cond      | Ordinal    | object  | Evaluates the present condition of the material on the exterior        |
| foundation      | Nominal    | object  | Type of foundation                                                     |
| bsmt_qual       | Ordinal    | object  | Evaluates the height of the basement                                   |
| bsmt_cond       | Ordinal    | object  | Evaluates the general condition of the basement                        |
| bsmt_exposure   | Ordinal    | object  | Refers to walkout or garden level walls                                |
| bsmtfin_type_1  | Ordinal    | object  | Rating of basement finished area                                       |
| bsmtfin_sf_1    | Continuous | float64 | Type 1 finished square feet                                            |
| bsmtfin_type_2  | Ordinal    | object  | Rating of basement finished area (if multiple types)                   |
| bsmtfin_sf_2    | Continuous | float64 | Type 2 finished square feet                                            |
| bsmt_unf_sf     | Continuous | float64 | Unfinished square feet of basement area                                |
| total_bsmt_sf   | Continuous | float64 | Total square feet of basement area                                     |
| heating         | Nominal    | object  | Type of heating                                                        |
| heating_qc      | Ordinal    | object  | Heating quality and condition                                          |
| central_air     | Nominal    | object  | Central air conditioning                                               |
| electrical      | Ordinal    | object  | Electrical system                                                      |
| 1st_flr_sf      | Continuous | int64   | First Floor square feet                                                |
| 2nd_flr_sf      | Continuous | int64   | Second floor square feet                                               |
| low_qual_fin_sf | Continuous | int64   | Low quality finished square feet (all floors)                          |
| gr_liv_area     | Continuous | int64   | Above grade (ground) living area square feet                           |
| bsmt_full_bath  | Discrete   | float64 | Basement full bathrooms                                                |
| bsmt_half_bath  | Discrete   | float64 | Basement half bathrooms                                                |
| full_bath       | Discrete   | int64   | Full bathrooms above grade                                             |
| half_bath       | Discrete   | int64   | Half baths above grade                                                 |
| bedroom_abvgr   | Discrete   | int64   | Bedrooms above grade (does NOT include basement bedrooms)              |
| kitchen_abvgr   | Discrete   | int64   | Kitchens above grade                                                   |
| kitchen_qual    | Ordinal    | object  | Kitchen quality                                                        |
| totrms_abvgrd   | Discrete   | int64   | Total rooms above grade (does not include bathrooms)                   |
| functional      | Ordinal    | object  | Home functionality (Assume typical unless deductions are warranted)    |
| fireplaces      | Discrete   | int64   | Number of fireplaces                                                   |
| fireplace_qu    | Ordinal    | object  | Fireplace quality                                                      |
| garage_type     | Nominal    | object  | Garage location                                                        |
| garage_yr_blt   | Discrete   | float64 | Year garage was built                                                  |
| garage_finish   | Ordinal    | object  | Interior finish of the garage                                          |
| garage_cars     | Discrete   | float64 | Size of garage in car capacity                                         |
| garage_area     | Continuous | float64 | Size of garage in square feet                                          |
| garage_qual     | Ordinal    | object  | Garage quality                                                         |
| garage_cond     | Ordinal    | object  | Garage condition                                                       |
| paved_drive     | Ordinal    | object  | Paved driveway                                                         |
| wood_deck_sf    | Continuous | int64   | Wood deck area in square feet                                          |
| open_porch_sf   | Continuous | int64   | Open porch area in square feet                                         |
| enclosed_porch  | Continuous | int64   | Enclosed porch area in square feet                                     |
| 3ssn_porch      | Continuous | int64   | Three season porch area in square feet                                 |
| screen_porch    | Continuous | int64   | Screen porch area in square feet                                       |
| pool_area       | Continuous | int64   | Pool area in square feet                                               |
| pool_qc         | Ordinal    | object  | Pool quality                                                           |
| fence           | Ordinal    | object  | Fence quality                                                          |
| misc_feature    | Nominal    | object  | Miscellaneous feature not covered in other categories                  |
| misc_val        | Continuous | int64   | Dollar value of miscellaneous feature                                  |
| mo_sold         | Discrete   | int64   | Month Sold (MM)                                                        |
| yr_sold         | Discrete   | int64   | Year Sold (YYYY)                                                       |
| sale_type       | Nominal    | object  | Type of sale                                                           |
| saleprice       | Continuous | int64   | Sale price                                                             |
### Engineered Features
| column_name           | data_type  | dtype | data_description                                                                                                |
|-----------------------|------------|-------|-----------------------------------------------------------------------------------------------------------------|
| eng_total_indoor_sqft | Continuous | int64 | The sum of `1st_floor_sq`, `2nd_floor_sf`, `basement_sf1`, `basement_sf2`, `basement_unf_sf`, and `garage_area` |
| eng_num_baths         | Discrete   | int64 | The sum of `full_bath` and `bsmt_full_bath` added to one-half the sum of `half_bath` and `bsmt_half_bath`       |
| eng_total_rooms       | Discrete   | int64 | The sum of `total_rooms_abv_grade` and `num_baths`                                                              |
| eng_remodel_x_built   | Continuous | int64 | The product of `year_remod_add` and `year_built`                                                                |


## Summary
The dataset comes from the City Assessor's office in Ames, Iowa [link](https://www.cityofames.org/government/departments-divisions-a-h/city-assessor) and consists of 2051 homes with 80 descriptive features. The features are recorded for the purpose of tax assessment. While this means they are generally quite comprehensive, there are some issues with the features which make them difficult to analyze for the purpose of prediction. Several discontinuites in the encoding system used requires the engineering of new features to prepare the data for analysis.

With proper encoding and imputation I was able to build an ordinary linear model with 121 features that was able to account for ~90% of the variance in sale price with a root mean square error of 22,855.

## Conclusions and recommendations
Features which contribute most to a homes price included overall size, and measures of overall condition and quality. Neighboorhood also had a very strong influence with neighborhoods like Green Hills, Stone Brook, and Northridge Heights had strong positive effects on predicted sale prices whereas Northwest Ames, Sawyer West, and College Creek had strong negative effects on predicted home prices. There are still some features in the dataset which could be included and regularization of the model would also likely include the fit.

For future work I would recommend looking for more than property assessor data. While it seems to be very comprehensize, it does have blind spots such as basement bedrooms not being recorded and lack of features like yard size, trees on the property, etc. Precise geolocation data would also be helpful. Going into the future additional metrics such as solar panels or home battery backup or generator systems should probably be included, as well as measures of energy efficiency. That being said, the model works very well even with the data we have.

## File structure
* README.md (this file)
* project-2.ipynb (a Jupyter notebook with a detailed rundown of the analysis)
* project-2-presentation.pdf (presentation slides summarizing this analysis)
* ./datasets
    - DataDocumentation.txt (original file supplied with dataset with comprehensive desciption of features.
    - feature_info.csv (a csv file of the key elements from DataDocumentation.txt used the classify features in the analysis)
    - train.csv (the dataset itself, used to train models)
    - test.csv (another dataset lacking sale price data, used for submitting predictions to the Kaggle competition.
    - sample_sub_reg.csv (an example submission file for Kaggle)
* ./submissions
    - (prediction files submitted to Kaggle)
    
## Sources
[Iowa Dept. of Revenue - Property Tax Overview](https://tax.iowa.gov/iowa-property-tax-overview)<br>
[Iowa Real Property Appraisal Manual](http://publications.iowa.gov/32396/)<br>
[221 Ames Competition](https://www.kaggle.com/competitions/221-ames-competition/)
