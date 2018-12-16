# coursera-practical-machine-learning

---
title: "Practical machine learning"
author: "Menno Bremer"
date: "9-12-2018"
output:
  html_document: https://github.com/mennobremer93/coursera-practical-machine-learning.git
  pdf_document: default
---

# Practical Machine Learning project

### Menno Bremer
### December 9, 2018

## Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

## Data

The training data for this project are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

## Load data

```{r setup, include=FALSE}
setwd("~/Documents/Coursera")
install.packages("caret")
install.packages("ggplot2")
install.packages("randomForest")
```

```{r}
library("caret")
library("ggplot2")
library("randomForest")

```

```{r}
train <- read.csv("pml-training.csv", na.strings=c("NA","#DIV/0!",""))
test <- read.csv("pml-testing.csv", na.strings=c("NA","#DIV/0!",""))

summary(train)
summary(train$classe)
```

    X            user_name    raw_timestamp_part_1 raw_timestamp_part_2          cvtd_timestamp  new_window    num_window   
 Min.   :    1   adelmo  :3892   Min.   :1.322e+09    Min.   :   294       28/11/2011 14:14: 1498   no :19216   Min.   :  1.0  
 1st Qu.: 4906   carlitos:3112   1st Qu.:1.323e+09    1st Qu.:252912       05/12/2011 11:24: 1497   yes:  406   1st Qu.:222.0  
 Median : 9812   charles :3536   Median :1.323e+09    Median :496380       30/11/2011 17:11: 1440               Median :424.0  
 Mean   : 9812   eurico  :3070   Mean   :1.323e+09    Mean   :500656       05/12/2011 11:25: 1425               Mean   :430.6  
 3rd Qu.:14717   jeremy  :3402   3rd Qu.:1.323e+09    3rd Qu.:751891       02/12/2011 14:57: 1380               3rd Qu.:644.0  
 Max.   :19622   pedro   :2610   Max.   :1.323e+09    Max.   :998801       02/12/2011 13:34: 1375               Max.   :864.0  
   roll_belt        pitch_belt          yaw_belt       total_accel_belt kurtosis_roll_belt kurtosis_picth_belt kurtosis_yaw_belt
 Min.   :-28.90   Min.   :-55.8000   Min.   :-180.00   Min.   : 0.00    Min.   :-2.121     Min.   :-2.190      Mode:logical     
 1st Qu.:  1.10   1st Qu.:  1.7600   1st Qu.: -88.30   1st Qu.: 3.00    1st Qu.:-1.329     1st Qu.:-1.107      NA's:19622       
 Median :113.00   Median :  5.2800   Median : -13.00   Median :17.00    Median :-0.899     Median :-0.151                       
 Mean   : 64.41   Mean   :  0.3053   Mean   : -11.21   Mean   :11.31    Mean   :-0.220     Mean   : 4.334                       
 3rd Qu.:123.00   3rd Qu.: 14.9000   3rd Qu.:  12.90   3rd Qu.:18.00    3rd Qu.:-0.219     3rd Qu.: 3.178                       
 Max.   :162.00   Max.   : 60.3000   Max.   : 179.00   Max.   :29.00    Max.   :33.000     Max.   :58.000                       
 skewness_roll_belt skewness_roll_belt.1 skewness_yaw_belt max_roll_belt     max_picth_belt   max_yaw_belt   min_roll_belt    
 Min.   :-5.745     Min.   :-7.616       Mode:logical      Min.   :-94.300   Min.   : 3.00   Min.   :-2.10   Min.   :-180.00  
 1st Qu.:-0.444     1st Qu.:-1.114       NA's:19622        1st Qu.:-88.000   1st Qu.: 5.00   1st Qu.:-1.30   1st Qu.: -88.40  
 Median : 0.000     Median :-0.068                         Median : -5.100   Median :18.00   Median :-0.90   Median :  -7.85  
 Mean   :-0.026     Mean   :-0.296                         Mean   : -6.667   Mean   :12.92   Mean   :-0.22   Mean   : -10.44  
 3rd Qu.: 0.417     3rd Qu.: 0.661                         3rd Qu.: 18.500   3rd Qu.:19.00   3rd Qu.:-0.20   3rd Qu.:   9.05  
 Max.   : 3.595     Max.   : 7.348                         Max.   :180.000   Max.   :30.00   Max.   :33.00   Max.   : 173.00  
 min_pitch_belt   min_yaw_belt   amplitude_roll_belt amplitude_pitch_belt amplitude_yaw_belt var_total_accel_belt avg_roll_belt   
 Min.   : 0.00   Min.   :-2.10   Min.   :  0.000     Min.   : 0.000       Min.   :0          Min.   : 0.000       Min.   :-27.40  
 1st Qu.: 3.00   1st Qu.:-1.30   1st Qu.:  0.300     1st Qu.: 1.000       1st Qu.:0          1st Qu.: 0.100       1st Qu.:  1.10  
 Median :16.00   Median :-0.90   Median :  1.000     Median : 1.000       Median :0          Median : 0.200       Median :116.35  
 Mean   :10.76   Mean   :-0.22   Mean   :  3.769     Mean   : 2.167       Mean   :0          Mean   : 0.926       Mean   : 68.06  
 3rd Qu.:17.00   3rd Qu.:-0.20   3rd Qu.:  2.083     3rd Qu.: 2.000       3rd Qu.:0          3rd Qu.: 0.300       3rd Qu.:123.38  
 Max.   :23.00   Max.   :33.00   Max.   :360.000     Max.   :12.000       Max.   :0          Max.   :16.500       Max.   :157.40  
 stddev_roll_belt var_roll_belt     avg_pitch_belt    stddev_pitch_belt var_pitch_belt    avg_yaw_belt      stddev_yaw_belt  
 Min.   : 0.000   Min.   :  0.000   Min.   :-51.400   Min.   :0.000     Min.   : 0.000   Min.   :-138.300   Min.   :  0.000  
 1st Qu.: 0.200   1st Qu.:  0.000   1st Qu.:  2.025   1st Qu.:0.200     1st Qu.: 0.000   1st Qu.: -88.175   1st Qu.:  0.100  
 Median : 0.400   Median :  0.100   Median :  5.200   Median :0.400     Median : 0.100   Median :  -6.550   Median :  0.300  
 Mean   : 1.337   Mean   :  7.699   Mean   :  0.520   Mean   :0.603     Mean   : 0.766   Mean   :  -8.831   Mean   :  1.341  
 3rd Qu.: 0.700   3rd Qu.:  0.500   3rd Qu.: 15.775   3rd Qu.:0.700     3rd Qu.: 0.500   3rd Qu.:  14.125   3rd Qu.:  0.700  
 Max.   :14.200   Max.   :200.700   Max.   : 59.700   Max.   :4.000     Max.   :16.200   Max.   : 173.500   Max.   :176.600  
  var_yaw_belt        gyros_belt_x        gyros_belt_y       gyros_belt_z      accel_belt_x       accel_belt_y     accel_belt_z    
 Min.   :    0.000   Min.   :-1.040000   Min.   :-0.64000   Min.   :-1.4600   Min.   :-120.000   Min.   :-69.00   Min.   :-275.00  
 1st Qu.:    0.010   1st Qu.:-0.030000   1st Qu.: 0.00000   1st Qu.:-0.2000   1st Qu.: -21.000   1st Qu.:  3.00   1st Qu.:-162.00  
 Median :    0.090   Median : 0.030000   Median : 0.02000   Median :-0.1000   Median : -15.000   Median : 35.00   Median :-152.00  
 Mean   :  107.487   Mean   :-0.005592   Mean   : 0.03959   Mean   :-0.1305   Mean   :  -5.595   Mean   : 30.15   Mean   : -72.59  
 3rd Qu.:    0.475   3rd Qu.: 0.110000   3rd Qu.: 0.11000   3rd Qu.:-0.0200   3rd Qu.:  -5.000   3rd Qu.: 61.00   3rd Qu.:  27.00  
 Max.   :31183.240   Max.   : 2.220000   Max.   : 0.64000   Max.   : 1.6200   Max.   :  85.000   Max.   :164.00   Max.   : 105.00  
 magnet_belt_x   magnet_belt_y   magnet_belt_z       roll_arm         pitch_arm          yaw_arm          total_accel_arm var_accel_arm   
 Min.   :-52.0   Min.   :354.0   Min.   :-623.0   Min.   :-180.00   Min.   :-88.800   Min.   :-180.0000   Min.   : 1.00   Min.   :  0.00  
 1st Qu.:  9.0   1st Qu.:581.0   1st Qu.:-375.0   1st Qu.: -31.77   1st Qu.:-25.900   1st Qu.: -43.1000   1st Qu.:17.00   1st Qu.:  9.03  
 Median : 35.0   Median :601.0   Median :-320.0   Median :   0.00   Median :  0.000   Median :   0.0000   Median :27.00   Median : 40.61  
 Mean   : 55.6   Mean   :593.7   Mean   :-345.5   Mean   :  17.83   Mean   : -4.612   Mean   :  -0.6188   Mean   :25.51   Mean   : 53.23  
 3rd Qu.: 59.0   3rd Qu.:610.0   3rd Qu.:-306.0   3rd Qu.:  77.30   3rd Qu.: 11.200   3rd Qu.:  45.8750   3rd Qu.:33.00   3rd Qu.: 75.62  
 Max.   :485.0   Max.   :673.0   Max.   : 293.0   Max.   : 180.00   Max.   : 88.500   Max.   : 180.0000   Max.   :66.00   Max.   :331.70  
  avg_roll_arm     stddev_roll_arm    var_roll_arm       avg_pitch_arm     stddev_pitch_arm var_pitch_arm       avg_yaw_arm      
 Min.   :-166.67   Min.   :  0.000   Min.   :    0.000   Min.   :-81.773   Min.   : 0.000   Min.   :   0.000   Min.   :-173.440  
 1st Qu.: -38.37   1st Qu.:  1.376   1st Qu.:    1.898   1st Qu.:-22.770   1st Qu.: 1.642   1st Qu.:   2.697   1st Qu.: -29.198  
 Median :   0.00   Median :  5.702   Median :   32.517   Median :  0.000   Median : 8.133   Median :  66.146   Median :   0.000  
 Mean   :  12.68   Mean   : 11.201   Mean   :  417.264   Mean   : -4.901   Mean   :10.383   Mean   : 195.864   Mean   :   2.359  
 3rd Qu.:  76.33   3rd Qu.: 14.921   3rd Qu.:  222.647   3rd Qu.:  8.277   3rd Qu.:16.327   3rd Qu.: 266.576   3rd Qu.:  38.185  
 Max.   : 163.33   Max.   :161.964   Max.   :26232.208   Max.   : 75.659   Max.   :43.412   Max.   :1884.565   Max.   : 152.000  
 stddev_yaw_arm     var_yaw_arm         gyros_arm_x        gyros_arm_y       gyros_arm_z       accel_arm_x       accel_arm_y    
 Min.   :  0.000   Min.   :    0.000   Min.   :-6.37000   Min.   :-3.4400   Min.   :-2.3300   Min.   :-404.00   Min.   :-318.0  
 1st Qu.:  2.577   1st Qu.:    6.642   1st Qu.:-1.33000   1st Qu.:-0.8000   1st Qu.:-0.0700   1st Qu.:-242.00   1st Qu.: -54.0  
 Median : 16.682   Median :  278.309   Median : 0.08000   Median :-0.2400   Median : 0.2300   Median : -44.00   Median :  14.0  
 Mean   : 22.270   Mean   : 1055.933   Mean   : 0.04277   Mean   :-0.2571   Mean   : 0.2695   Mean   : -60.24   Mean   :  32.6  
 3rd Qu.: 35.984   3rd Qu.: 1294.850   3rd Qu.: 1.57000   3rd Qu.: 0.1400   3rd Qu.: 0.7200   3rd Qu.:  84.00   3rd Qu.: 139.0  
 Max.   :177.044   Max.   :31344.568   Max.   : 4.87000   Max.   : 2.8400   Max.   : 3.0200   Max.   : 437.00   Max.   : 308.0  
  accel_arm_z       magnet_arm_x     magnet_arm_y     magnet_arm_z    kurtosis_roll_arm kurtosis_picth_arm kurtosis_yaw_arm
 Min.   :-636.00   Min.   :-584.0   Min.   :-392.0   Min.   :-597.0   Min.   :-1.809    Min.   :-2.084     Min.   :-2.103  
 1st Qu.:-143.00   1st Qu.:-300.0   1st Qu.:  -9.0   1st Qu.: 131.2   1st Qu.:-1.345    1st Qu.:-1.280     1st Qu.:-1.220  
 Median : -47.00   Median : 289.0   Median : 202.0   Median : 444.0   Median :-0.894    Median :-1.010     Median :-0.733  
 Mean   : -71.25   Mean   : 191.7   Mean   : 156.6   Mean   : 306.5   Mean   :-0.366    Mean   :-0.542     Mean   : 0.406  
 3rd Qu.:  23.00   3rd Qu.: 637.0   3rd Qu.: 323.0   3rd Qu.: 545.0   3rd Qu.:-0.038    3rd Qu.:-0.379     3rd Qu.: 0.115  
 Max.   : 292.00   Max.   : 782.0   Max.   : 583.0   Max.   : 694.0   Max.   :21.456    Max.   :19.751     Max.   :56.000  
 skewness_roll_arm skewness_pitch_arm skewness_yaw_arm  max_roll_arm     max_picth_arm       max_yaw_arm     min_roll_arm   
 Min.   :-2.541    Min.   :-4.565     Min.   :-6.708   Min.   :-73.100   Min.   :-173.000   Min.   : 4.00   Min.   :-89.10  
 1st Qu.:-0.561    1st Qu.:-0.618     1st Qu.:-0.743   1st Qu.: -0.175   1st Qu.:  -1.975   1st Qu.:29.00   1st Qu.:-41.98  
 Median : 0.040    Median :-0.035     Median :-0.133   Median :  4.950   Median :  23.250   Median :34.00   Median :-22.45  
 Mean   : 0.068    Mean   :-0.065     Mean   :-0.229   Mean   : 11.236   Mean   :  35.751   Mean   :35.46   Mean   :-21.22  
 3rd Qu.: 0.671    3rd Qu.: 0.454     3rd Qu.: 0.344   3rd Qu.: 26.775   3rd Qu.:  95.975   3rd Qu.:41.00   3rd Qu.:  0.00  
 Max.   : 4.394    Max.   : 3.043     Max.   : 7.483   Max.   : 85.500   Max.   : 180.000   Max.   :65.00   Max.   : 66.40  
 min_pitch_arm      min_yaw_arm    amplitude_roll_arm amplitude_pitch_arm amplitude_yaw_arm roll_dumbbell     pitch_dumbbell   
 Min.   :-180.00   Min.   : 1.00   Min.   :  0.000    Min.   :  0.000     Min.   : 0.00     Min.   :-153.71   Min.   :-149.59  
 1st Qu.: -72.62   1st Qu.: 8.00   1st Qu.:  5.425    1st Qu.:  9.925     1st Qu.:13.00     1st Qu.: -18.49   1st Qu.: -40.89  
 Median : -33.85   Median :13.00   Median : 28.450    Median : 54.900     Median :22.00     Median :  48.17   Median : -20.96  
 Mean   : -33.92   Mean   :14.66   Mean   : 32.452    Mean   : 69.677     Mean   :20.79     Mean   :  23.84   Mean   : -10.78  
 3rd Qu.:   0.00   3rd Qu.:19.00   3rd Qu.: 50.960    3rd Qu.:115.175     3rd Qu.:28.75     3rd Qu.:  67.61   3rd Qu.:  17.50  
 Max.   : 152.00   Max.   :38.00   Max.   :119.500    Max.   :360.000     Max.   :52.00     Max.   : 153.55   Max.   : 149.40  
  yaw_dumbbell      kurtosis_roll_dumbbell kurtosis_picth_dumbbell kurtosis_yaw_dumbbell skewness_roll_dumbbell skewness_pitch_dumbbell
 Min.   :-150.871   Min.   :-2.174         Min.   :-2.200          Mode:logical          Min.   :-7.384         Min.   :-7.447         
 1st Qu.: -77.644   1st Qu.:-0.682         1st Qu.:-0.721          NA's:19622            1st Qu.:-0.581         1st Qu.:-0.526         
 Median :  -3.324   Median :-0.033         Median :-0.133                                Median :-0.076         Median :-0.091         
 Mean   :   1.674   Mean   : 0.452         Mean   : 0.286                                Mean   :-0.115         Mean   :-0.035         
 3rd Qu.:  79.643   3rd Qu.: 0.940         3rd Qu.: 0.584                                3rd Qu.: 0.400         3rd Qu.: 0.505         
 Max.   : 154.952   Max.   :54.998         Max.   :55.628                                Max.   : 1.958         Max.   : 3.769         
 skewness_yaw_dumbbell max_roll_dumbbell max_picth_dumbbell max_yaw_dumbbell min_roll_dumbbell min_pitch_dumbbell min_yaw_dumbbell
 Mode:logical          Min.   :-70.10    Min.   :-112.90    Min.   :-2.20    Min.   :-149.60   Min.   :-147.00    Min.   :-2.20   
 NA's:19622            1st Qu.:-27.15    1st Qu.: -66.70    1st Qu.:-0.70    1st Qu.: -59.67   1st Qu.: -91.80    1st Qu.:-0.70   
                       Median : 14.85    Median :  40.05    Median : 0.00    Median : -43.55   Median : -66.15    Median : 0.00   
                       Mean   : 13.76    Mean   :  32.75    Mean   : 0.45    Mean   : -41.24   Mean   : -33.18    Mean   : 0.45   
                       3rd Qu.: 50.58    3rd Qu.: 133.22    3rd Qu.: 0.90    3rd Qu.: -25.20   3rd Qu.:  21.20    3rd Qu.: 0.90   
                       Max.   :137.00    Max.   : 155.00    Max.   :55.00    Max.   :  73.20   Max.   : 120.90    Max.   :55.00   
 amplitude_roll_dumbbell amplitude_pitch_dumbbell amplitude_yaw_dumbbell total_accel_dumbbell var_accel_dumbbell avg_roll_dumbbell
 Min.   :  0.00          Min.   :  0.00           Min.   :0              Min.   : 0.00        Min.   :  0.000    Min.   :-128.96  
 1st Qu.: 14.97          1st Qu.: 17.06           1st Qu.:0              1st Qu.: 4.00        1st Qu.:  0.378    1st Qu.: -12.33  
 Median : 35.05          Median : 41.73           Median :0              Median :10.00        Median :  1.000    Median :  48.23  
 Mean   : 55.00          Mean   : 65.93           Mean   :0              Mean   :13.72        Mean   :  4.388    Mean   :  23.86  
 3rd Qu.: 81.04          3rd Qu.: 99.55           3rd Qu.:0              3rd Qu.:19.00        3rd Qu.:  3.434    3rd Qu.:  64.37  
 Max.   :256.48          Max.   :273.59           Max.   :0              Max.   :58.00        Max.   :230.428    Max.   : 125.99  
 stddev_roll_dumbbell var_roll_dumbbell  avg_pitch_dumbbell stddev_pitch_dumbbell var_pitch_dumbbell avg_yaw_dumbbell   stddev_yaw_dumbbell
 Min.   :  0.000      Min.   :    0.00   Min.   :-70.73     Min.   : 0.000        Min.   :   0.00    Min.   :-117.950   Min.   :  0.000    
 1st Qu.:  4.639      1st Qu.:   21.52   1st Qu.:-42.00     1st Qu.: 3.482        1st Qu.:  12.12    1st Qu.: -76.696   1st Qu.:  3.885    
 Median : 12.204      Median :  148.95   Median :-19.91     Median : 8.089        Median :  65.44    Median :  -4.505   Median : 10.264    
 Mean   : 20.761      Mean   : 1020.27   Mean   :-12.33     Mean   :13.147        Mean   : 350.31    Mean   :   0.202   Mean   : 16.647    
 3rd Qu.: 26.356      3rd Qu.:  694.65   3rd Qu.: 13.21     3rd Qu.:19.238        3rd Qu.: 370.11    3rd Qu.:  71.234   3rd Qu.: 24.674    
 Max.   :123.778      Max.   :15321.01   Max.   : 94.28     Max.   :82.680        Max.   :6836.02    Max.   : 134.905   Max.   :107.088    
 var_yaw_dumbbell   gyros_dumbbell_x    gyros_dumbbell_y   gyros_dumbbell_z  accel_dumbbell_x  accel_dumbbell_y  accel_dumbbell_z 
 Min.   :    0.00   Min.   :-204.0000   Min.   :-2.10000   Min.   : -2.380   Min.   :-419.00   Min.   :-189.00   Min.   :-334.00  
 1st Qu.:   15.09   1st Qu.:  -0.0300   1st Qu.:-0.14000   1st Qu.: -0.310   1st Qu.: -50.00   1st Qu.:  -8.00   1st Qu.:-142.00  
 Median :  105.35   Median :   0.1300   Median : 0.03000   Median : -0.130   Median :  -8.00   Median :  41.50   Median :  -1.00  
 Mean   :  589.84   Mean   :   0.1611   Mean   : 0.04606   Mean   : -0.129   Mean   : -28.62   Mean   :  52.63   Mean   : -38.32  
 3rd Qu.:  608.79   3rd Qu.:   0.3500   3rd Qu.: 0.21000   3rd Qu.:  0.030   3rd Qu.:  11.00   3rd Qu.: 111.00   3rd Qu.:  38.00  
 Max.   :11467.91   Max.   :   2.2200   Max.   :52.00000   Max.   :317.000   Max.   : 235.00   Max.   : 315.00   Max.   : 318.00  
 magnet_dumbbell_x magnet_dumbbell_y magnet_dumbbell_z  roll_forearm       pitch_forearm     yaw_forearm      kurtosis_roll_forearm
 Min.   :-643.0    Min.   :-3600     Min.   :-262.00   Min.   :-180.0000   Min.   :-72.50   Min.   :-180.00   Min.   :-1.879       
 1st Qu.:-535.0    1st Qu.:  231     1st Qu.: -45.00   1st Qu.:  -0.7375   1st Qu.:  0.00   1st Qu.: -68.60   1st Qu.:-1.398       
 Median :-479.0    Median :  311     Median :  13.00   Median :  21.7000   Median :  9.24   Median :   0.00   Median :-1.119       
 Mean   :-328.5    Mean   :  221     Mean   :  46.05   Mean   :  33.8265   Mean   : 10.71   Mean   :  19.21   Mean   :-0.689       
 3rd Qu.:-304.0    3rd Qu.:  390     3rd Qu.:  95.00   3rd Qu.: 140.0000   3rd Qu.: 28.40   3rd Qu.: 110.00   3rd Qu.:-0.618       
 Max.   : 592.0    Max.   :  633     Max.   : 452.00   Max.   : 180.0000   Max.   : 89.80   Max.   : 180.00   Max.   :40.060       
 kurtosis_picth_forearm kurtosis_yaw_forearm skewness_roll_forearm skewness_pitch_forearm skewness_yaw_forearm max_roll_forearm
 Min.   :-2.098         Mode:logical         Min.   :-2.297        Min.   :-5.241         Mode:logical         Min.   :-66.60  
 1st Qu.:-1.376         NA's:19622           1st Qu.:-0.402        1st Qu.:-0.881         NA's:19622           1st Qu.:  0.00  
 Median :-0.890                              Median : 0.003        Median :-0.156                              Median : 26.80  
 Mean   : 0.419                              Mean   :-0.009        Mean   :-0.223                              Mean   : 24.49  
 3rd Qu.: 0.054                              3rd Qu.: 0.370        3rd Qu.: 0.514                              3rd Qu.: 45.95  
 Max.   :33.626                              Max.   : 5.856        Max.   : 4.464                              Max.   : 89.80  
 max_picth_forearm max_yaw_forearm  min_roll_forearm  min_pitch_forearm min_yaw_forearm  amplitude_roll_forearm amplitude_pitch_forearm
 Min.   :-151.00   Min.   :-1.900   Min.   :-72.500   Min.   :-180.00   Min.   :-1.900   Min.   :  0.000        Min.   :  0.0          
 1st Qu.:   0.00   1st Qu.:-1.400   1st Qu.: -6.075   1st Qu.:-175.00   1st Qu.:-1.400   1st Qu.:  1.125        1st Qu.:  2.0          
 Median : 113.00   Median :-1.100   Median :  0.000   Median : -61.00   Median :-1.100   Median : 17.770        Median : 83.7          
 Mean   :  81.49   Mean   :-0.689   Mean   : -0.167   Mean   : -57.57   Mean   :-0.689   Mean   : 24.653        Mean   :139.1          
 3rd Qu.: 174.75   3rd Qu.:-0.600   3rd Qu.: 12.075   3rd Qu.:   0.00   3rd Qu.:-0.600   3rd Qu.: 39.875        3rd Qu.:350.0          
 Max.   : 180.00   Max.   :40.100   Max.   : 62.100   Max.   : 167.00   Max.   :40.100   Max.   :126.000        Max.   :360.0          
 amplitude_yaw_forearm total_accel_forearm var_accel_forearm avg_roll_forearm   stddev_roll_forearm var_roll_forearm   avg_pitch_forearm
 Min.   :0             Min.   :  0.00      Min.   :  0.000   Min.   :-177.234   Min.   :  0.000     Min.   :    0.00   Min.   :-68.17   
 1st Qu.:0             1st Qu.: 29.00      1st Qu.:  6.759   1st Qu.:  -0.909   1st Qu.:  0.428     1st Qu.:    0.18   1st Qu.:  0.00   
 Median :0             Median : 36.00      Median : 21.165   Median :  11.172   Median :  8.030     Median :   64.48   Median : 12.02   
 Mean   :0             Mean   : 34.72      Mean   : 33.502   Mean   :  33.165   Mean   : 41.986     Mean   : 5274.10   Mean   : 11.79   
 3rd Qu.:0             3rd Qu.: 41.00      3rd Qu.: 51.240   3rd Qu.: 107.132   3rd Qu.: 85.373     3rd Qu.: 7289.08   3rd Qu.: 28.48   
 Max.   :0             Max.   :108.00      Max.   :172.606   Max.   : 177.256   Max.   :179.171     Max.   :32102.24   Max.   : 72.09   
 stddev_pitch_forearm var_pitch_forearm  avg_yaw_forearm   stddev_yaw_forearm var_yaw_forearm    gyros_forearm_x   gyros_forearm_y    
 Min.   : 0.000       Min.   :   0.000   Min.   :-155.06   Min.   :  0.000    Min.   :    0.00   Min.   :-22.000   Min.   : -7.02000  
 1st Qu.: 0.336       1st Qu.:   0.113   1st Qu.: -26.26   1st Qu.:  0.524    1st Qu.:    0.27   1st Qu.: -0.220   1st Qu.: -1.46000  
 Median : 5.516       Median :  30.425   Median :   0.00   Median : 24.743    Median :  612.21   Median :  0.050   Median :  0.03000  
 Mean   : 7.977       Mean   : 139.593   Mean   :  18.00   Mean   : 44.854    Mean   : 4639.85   Mean   :  0.158   Mean   :  0.07517  
 3rd Qu.:12.866       3rd Qu.: 165.532   3rd Qu.:  85.79   3rd Qu.: 85.817    3rd Qu.: 7368.41   3rd Qu.:  0.560   3rd Qu.:  1.62000  
 Max.   :47.745       Max.   :2279.617   Max.   : 169.24   Max.   :197.508    Max.   :39009.33   Max.   :  3.970   Max.   :311.00000  
 gyros_forearm_z    accel_forearm_x   accel_forearm_y  accel_forearm_z   magnet_forearm_x  magnet_forearm_y magnet_forearm_z classe  
 Min.   : -8.0900   Min.   :-498.00   Min.   :-632.0   Min.   :-446.00   Min.   :-1280.0   Min.   :-896.0   Min.   :-973.0   A:5580  
 1st Qu.: -0.1800   1st Qu.:-178.00   1st Qu.:  57.0   1st Qu.:-182.00   1st Qu.: -616.0   1st Qu.:   2.0   1st Qu.: 191.0   B:3797  
 Median :  0.0800   Median : -57.00   Median : 201.0   Median : -39.00   Median : -378.0   Median : 591.0   Median : 511.0   C:3422  
 Mean   :  0.1512   Mean   : -61.65   Mean   : 163.7   Mean   : -55.29   Mean   : -312.6   Mean   : 380.1   Mean   : 393.6   D:3216  
 3rd Qu.:  0.4900   3rd Qu.:  76.00   3rd Qu.: 312.0   3rd Qu.:  26.00   3rd Qu.:  -73.0   3rd Qu.: 737.0   3rd Qu.: 653.0   E:3607  
 Max.   :231.0000   Max.   : 477.00   Max.   : 923.0   Max.   : 291.00   Max.   :  672.0   Max.   :1480.0   Max.   :1090.0           
 [ reached getOption("max.print") -- omitted 1 row ]
   A    B    C    D    E 
5580 3797 3422 3216 3607 

## Split training and test data

Set aside a subset of the training data for cross validation (70%)

```{r}
Training <- createDataPartition(y=train$classe, p=0.7, list=FALSE)
inTrain <- train[Training, ]
inTest <- train[-Training, ]
dim(inTrain)
```
[1] 13737   160

```{r}
dim(inTest)
```

[1] 5885  160

## Feature Selection

Second, transform the data, include only the variables we want to use to built the prediction model. 
Remove variables with near zero variance, missing data, and variables which are useless as predictors.

```{r}
intrainselection <- inTrain
for (i in 1:length(inTrain)) {
  if (sum(is.na(inTrain[ , i])) / nrow(inTrain) >= .7) {
    for (j in 1:length(intrainselection)) {
      if (length(grep(names(inTrain[i]), names(intrainselection)[j]))==1) {
        intrainselection <- intrainselection[ , -j]
      }
    }
  }
}

dim(intrainselection)
```

[1] 13737    60

```{r}
#remove columns that are obviously not predictors
intrains <- intrainselection[,8:length(intrainselection)]

#remove variables with near zero variance
NZV <- nearZeroVar(intrains, saveMetrics = TRUE)
```

```{r}
keep <- names(intrains)

```

## Random Forest Model

Random forest model is used to build the Machine Learning algorithm as it should be more accurate than most other models based on information from the lectures.

First, fit the model on the training data.

```{r}
set.seed(151)

fitmod <- randomForest(classe~., data = intrains)
print(fitmod)
```

Call:
 randomForest(formula = classe ~ ., data = intrains) 
               Type of random forest: classification
                     Number of trees: 500
No. of variables tried at each split: 7

        OOB estimate of  error rate: 0.58%
Confusion matrix:
     A    B    C    D    E  class.error
A 3903    2    0    0    1 0.0007680492
B   15 2638    5    0    0 0.0075244545
C    0   16 2377    3    0 0.0079298831
D    0    0   24 2226    2 0.0115452931
E    0    0    1   10 2514 0.0043564356

### Out of sample error
Second, use the model to predict the variable classe on the subset of testing data (cross validation).

```{r}
install.packages("e1071")
```

```{r}
library("e1071")
```

```{r}
predict1 <- predict(fitmod, inTest, type = "class")
confusionMatrix(inTest$classe, predict1)
```
Confusion Matrix and Statistics

          Reference
Prediction    A    B    C    D    E
         A 1674    0    0    0    0
         B    4 1135    0    0    0
         C    0    5 1020    1    0
         D    0    0    8  956    0
         E    0    0    0    8 1074

Overall Statistics
                                          
               Accuracy : 0.9956          
                 95% CI : (0.9935, 0.9971)
    No Information Rate : 0.2851          
    P-Value [Acc > NIR] : < 2.2e-16       
                                          
                  Kappa : 0.9944          
 Mcnemar's Test P-Value : NA              

Statistics by Class:

                     Class: A Class: B Class: C Class: D Class: E
Sensitivity            0.9976   0.9956   0.9922   0.9907   1.0000
Specificity            1.0000   0.9992   0.9988   0.9984   0.9983
Pos Pred Value         1.0000   0.9965   0.9942   0.9917   0.9926
Neg Pred Value         0.9991   0.9989   0.9984   0.9982   1.0000
Prevalence             0.2851   0.1937   0.1747   0.1640   0.1825
Detection Rate         0.2845   0.1929   0.1733   0.1624   0.1825
Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
Balanced Accuracy      0.9988   0.9974   0.9955   0.9945   0.9992

### in sample error

```{r}
predicttrain <- predict(fitmod, inTrain, type = "class")
confusionMatrix(inTrain$classe, predicttrain)
```

Confusion Matrix and Statistics

          Reference
Prediction    A    B    C    D    E
         A 3906    0    0    0    0
         B    0 2658    0    0    0
         C    0    0 2396    0    0
         D    0    0    0 2252    0
         E    0    0    0    0 2525

Overall Statistics
                                     
               Accuracy : 1          
                 95% CI : (0.9997, 1)
    No Information Rate : 0.2843     
    P-Value [Acc > NIR] : < 2.2e-16  
                                     
                  Kappa : 1          
 Mcnemar's Test P-Value : NA         

Statistics by Class:

                     Class: A Class: B Class: C Class: D Class: E
Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
Prevalence             0.2843   0.1935   0.1744   0.1639   0.1838
Detection Rate         0.2843   0.1935   0.1744   0.1639   0.1838
Detection Prevalence   0.2843   0.1935   0.1744   0.1639   0.1838
Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000

Accuracy of in sample is lower than out of sample (99.63% vs 100%). 

## Model (test set)

```{r}
predict3 <- predict(fitmod, test, type = "class")
print(predict3)
```
1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
 B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
Levels: A B C D E

### Reference  
Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative 
Activity Recognition of Weight Lifting Exercises. Proceedings of 4th 
International Conference in Cooperation with SIGCHI (Augmented Human '13).
Stuttgart, Germany: ACM SIGCHI, 2013.

