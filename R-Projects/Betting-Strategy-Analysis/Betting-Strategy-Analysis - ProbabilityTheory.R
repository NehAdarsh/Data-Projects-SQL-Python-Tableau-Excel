###############Part 1#################

######### 1 #############

#Prbability of winning Red sox

#as we are calculating only winning probability of red sox   

#W : Win
#L : Lose

## BO   NY   BO  
------------------
  #1  W    W    -                
  #2  W    L    W              
  #3  L    W    W    
  ------------------
  
red_sox = 0.6 * 0.43 + 0.6 * 0.57 * 0.6 + 0.4 * 0.43 * 0.6  #0.5664


######### 2 ############

#Random variable : Net win (X)
#Red sox wins the first two games :
X = (0.6)*(1-0.57)   # Res : 0.258, X = 1000

#Red sox wins the 1st and 3rd game or 2nd and 3rd:
X = (0.6*0.57*0.6) + (1-0.6)*(1-0.57)*0.6   #res : 0.3084, X : 480

#Red sox win either first or second game:
X = 0.6*0.57*(1-0.6) + (1-0.6)*(1-0.57)*(1-0.6)   #res : 0.2056, X = -540

#Red sox loses both the games (lost the series)
X = (1-0.6) * 0.57      #res : 0.228 , X = -1040

# E[X] : Calculating the expected net win 
#-----------------------------------------
#   x     |     P(x)     | x*P(x)        |
#-----------------------------------------
#  1000   |    0.258     |  258          |
#-----------------------------------------
#  480    |   0.3084     |  148.032      |
#-----------------------------------------
# -540    |   0.2056     | -111.024      |
#-----------------------------------------
# -1040   |   0.228      | -237.12       |
#-----------------------------------------
#               E[X]:    |  57.888       |
#-----------------------------------------


#   Random values for X
set.seed(2)
p = c(0.258, 0.3084, 0.2056, 0.228)
x = c(1000, 480, -540, -1040)
Y = sample(x = x, size = 10000, replace = T, prob = p)

m = mean(Y)    #mean : 47.298
sd = sd(Y)     #Sd: 794.5653

#Confidence interval
CI = sort(Y)        
CI[0.025*10000]      # -1040
CI[0.975*10000]      # 1000

######### 3 ############

#Frequency Table
set.seed(1)
freq_table = table(Y)


#Chi-squared goodness of fit test

# Values:
#-----------------------------------------
#   Y     |     Observed     | Expected   |
#-----------------------------------------
#  1000   |       2523       |   2580     |
#-----------------------------------------
#  480    |       3078       |   3084     |
#-----------------------------------------
# -540    |       2095       |   2056     |
#-----------------------------------------
# -1040   |       2304       |   2280     |
#-----------------------------------------

#Null Hypothesis : There are no significant differences between Y distributions and X distributions
#Alternative Hypothesis : There are significant differences between Y distributions and X distributions
alpha = 0.05

ob = c(2523, 3078, 2095, 2304)
ex = c(0.258, 0.3084, 0.2056, 0.228)

chisq.test(x = ob, p = ex)   #P-value is 0.51 > 0.05, Fail to reject the null hypothesis.



#############################################################################################################################
#############################################################################################################################
##############################################################################################################################

###############Part 2#################

#Prbability of winning Red sox
red_sox_win = 0.43 * 0.6 + 0.43 * 0.43 * 0.4 + 0.4 * 0.43 * 0.6  # 0.43516

#as we are calculating only winning probability of red sox   

#W : Win
#L : Lose

##   NY   BO   NY  
------------------
  #1  W    W    -                
  #2  W    L    W              
  #3  L    W    W    
  ------------------

######### 2 ############

#Random variable : Net win (X)
#Red sox wins the first two games :
X = 0.43*0.6   # Res : 0.258, X = 1000

#Red sox wins the 1st and 3rd game or 2nd and 3rd game
X = (0.43 * 0.4 * 0.43) + (0.57 * 0.6 * 0.43) #res : 0.22102, X : 480

#Red sox win either first or second game:
X = (0.57*0.6*0.57) + (0.43*0.4*0.57)    #res : 0.29298, X = -540

#Red sox loses both the games (lost the series)
X = 0.57 * 0.4      #res : 0.228 , X = -1040

# E[X] : Calculating the expected net win 
#-----------------------------------------
#   x     |     P(x)     | x*P(x)        |
#-----------------------------------------
#  1000   |    0.258     |  258          |
#-----------------------------------------
#  480    |   0.22102    |  106.0896     |
#-----------------------------------------
# -540    |   0.29298    | -158.2092     |
#-----------------------------------------
# -1040   |   0.228      | -237.12       |
#-----------------------------------------
#               E[X]:    | -31.2396      |
#-----------------------------------------

#   Random values for X
set.seed(2)
p = c(0.258, 0.22102, 0.29298, 0.228)
x = c(1000, 480, -540, -1040)
Y = sample(x = x, size = 10000, replace = T, prob = p)

m = mean(Y)    #mean : -33.566
sd = sd(Y)     #Sd: 799.4217

#Confidence interval
CI = sort(Y)        
CI[0.025*10000]      # -1040
CI[0.975*10000]      # 1000


######### 3 ############

#Frequency Table
set.seed(1)
freq_table = table(Y)


#Chi-squared goodness of fit test
# Values:
#-----------------------------------------
#   Y     |     Observed     | Expected   |
#-----------------------------------------
#  1000   |       2549       |   2580     |
#-----------------------------------------
#  480    |       2244       |   22102    |
#-----------------------------------------
# -540    |       2907       |   29298    |
#-----------------------------------------
# -1040   |       2300       |   2280     |
#-----------------------------------------

#Null Hypothesis : There are no significant differences between Y distributions and X distributions
#Alternative Hypothesis : There are significant differences between Y distributions and X distributions
alpha = 0.05

ob = c(2549, 2244, 2907, 2300)
ex = c(0.258, 0.22102, 0.29298, 0.228)

chisq.test(x = ob, p = ex)   #P-value is 0.74 > 0.05, Fail to reject the null hypothesis.

#############################################################################################################################
#############################################################################################################################
##############################################################################################################################

###############  Part 3  #################

###### 1 ########

#Total combinations : 5C5 + 5C4 + 5C3 + 5C2 + 5C1 + 5C0 = 32
#WINNING : 5C3 : 10
#LOSING : 5C2 : 10 && 5C1 = 5 && 5C0 = 1

#Probability of winning Red sox in best of 5 game series

#   5C3 : 10 combinations_         # 5C2 : 10                         # 5C1 : 5                      #5C0
#  BO   NY   BO    NY   BO           #BO   NY   BO    NY   BO           #BO   NY   BO    NY   BO       #BO   NY   BO    NY   BO
#1  W    W    W    L    L            # L    L    L    W    W            # L    L    L    L    W        # L    L    L    L    L
#2  W    W    L    W    L            # L    L    W    L    W            # L    L    L    W    L
#3  W    L    W    W    L            # L    W    L    L    W            # L    L    W    L    L
#4  L    W    W    W    L            # W    L    L    L    W            # L    W    L    L    L
#5  W    W    L    L    W            # L    L    W    W    L            # W    L    L    L    L
#6  W    L    W    L    W            # L    W    L    W    L
#7  L    W    W    L    W            # W    L    L    W    L
#8  W    L    L    W    W            # L    W    W    L    L
#9  L    W    L    W    W            # W    L    W    L    L
#10  L    L    W    W    W           # W    W    L    L    L 


red_sox_win = 0.6*0.43*0.6 + 0.6 * 0.43 * 0.4 * 0.43 + 0.6 * 0.4 * 0.6 * 0.43 + 0.4*0.43*0.6*0.43+
  0.6*0.43*0.4*0.57*0.6 + 0.6*0.57*0.6*0.57*0.6 + 0.4*0.43*0.6*0.57*0.6 + 0.6*0.57*0.4*0.43*0.6 + 0.4*0.43*0.4*0.43*0.6+        # 0.5896944
  0.4*0.57*0.6*0.43*0.6

######### 2 ############

#Random variable : Net win (X)

# W : Win, L : Lose
red_Win_B = 0.6
red_Win_N = 0.43
ny_Win_B = 0.4
ny_Win_N = 0.57

#WWW :
X = red_Win_B*red_Win_N*red_Win_B   # Res : 0.1548, X = 1500

#WWLW  & #LWWW:
X = (red_Win_B * red_Win_N * ny_Win_B * red_Win_N) +  (ny_Win_B*red_Win_N*red_Win_B*red_Win_N)   #res : 0.088752, X : 980

#WLWW
X = red_Win_B * ny_Win_N * red_Win_B * red_Win_N    #res 0.088236, X = 980

#WWLLW
X = red_Win_B*red_Win_N*ny_Win_B*ny_Win_N*red_Win_B   #res 0.0352944, X = 980

#WLWLW
X = red_Win_B*ny_Win_N*red_Win_B*ny_Win_N*red_Win_B   #res 0.0701784, X = 460

#LWWLW   &   #WLLWW 
X = (ny_Win_B*red_Win_N*red_Win_B*ny_Win_N*red_Win_B) + (red_Win_B*ny_Win_N*ny_Win_B*red_Win_N*red_Win_B)  #res 0.0705888, X = 460

# LWLWW
X =  ny_Win_B*red_Win_N*ny_Win_B*red_Win_N*red_Win_B   #res  0.0177504, X = 460

# LLWWW 
X =  ny_Win_B*ny_Win_N*red_Win_B*red_Win_N*red_Win_B   #res  0.0352944, X = 460

#LLL
X =  ny_Win_B*ny_Win_N*ny_Win_B   #res  0.0912, X = -1560

#LLWL  && #WLLL
X = (ny_Win_B*ny_Win_N*red_Win_B*ny_Win_N) + (red_Win_B*ny_Win_N*ny_Win_B*ny_Win_N)  #res : 0.155952, X = -1060

#LWLL
X = ny_Win_B*red_Win_N*ny_Win_B*ny_Win_N   #res : 0.039216, X = -1060

#LLWWL  && #WLLWL && #LWWLL && #WWLLL
X = (ny_Win_B*ny_Win_N*red_Win_B*red_Win_N*ny_Win_B) + (red_Win_B*ny_Win_N*ny_Win_B*red_Win_N*ny_Win_B ) +
  (ny_Win_B*red_Win_N*red_Win_B*ny_Win_N*ny_Win_B) + (red_Win_B*red_Win_N*ny_Win_B*ny_Win_N*ny_Win_B) #res : 0.0941184, X = -560

#LWLWL
X = ny_Win_B*red_Win_N*ny_Win_B*red_Win_N*ny_Win_B  #res : 0.0118336, X = -560

#WLWLL
X = red_Win_B*ny_Win_N*red_Win_B*ny_Win_N*ny_Win_B  #res : 0.0467856, X = -560



# E[X] : Calculating the expected net win 
#-----------------------------------------
# x  | P(x)   |   x*P(x)     
#-----------------------------------------
1500*0.1548     # 232.2     
#-----------------------------------------
980*0.088752    # 86.97696  
#-----------------------------------------
980*0.088236    # 86.47128 
#-----------------------------------------
980*0.0352944   # 34.58851
#-----------------------------------------
460*0.0701784   # 32.28206
#-----------------------------------------
460*0.0705888   # 32.47085
#-----------------------------------------
460*0.0177504   # 8.165184
#-----------------------------------------
460*0.0352944   # 16.23542
#-----------------------------------------
-1560*0.0912    # -142.272
#-----------------------------------------
-1060*0.155952  # -165.3091
#-----------------------------------------
-1060*0.039216  # -41.56896
#-----------------------------------------
-560*0.0941184  # -52.7063
#-----------------------------------------
-560*0.0118336  # -6.626816
#-----------------------------------------
-560*0.0467856  # -26.19994
------------------------------------------
# E[X]:         |  94.70715   |
#-----------------------------------------


s = sum(0.1548,0.088752,0.088236,0.0352944,0.0701784,0.0705888,0.0177504,0.0352944,0.0912,0.155952,0.039216,0.0941184,0.0118336,0.0467856)
sum_exp = sum(232.2,86.97696 ,86.47128 ,34.58851 ,32.28206 ,32.47085 ,8.165184 ,16.23542,-41.56896, -52.7063,-165.3091,-142.272,-26.19994,-6.626816)


#   Random values for X
set.seed(2)
p = c(0.1548,0.088752,0.088236,0.0352944,0.0701784,0.0705888,0.0177504,0.0352944,0.0912,0.155952,0.039216,0.0941184,0.0118336,0.0467856)
x = c(1500,980,980,980,460,460,460,460,-1560,-1060,-1060,-560,-560,-560)
Y = sample(x = x, size = 10000, replace = T, prob = p)

m = mean(Y)    #mean : 101.246
sd = sd(Y)     #Sd: 1032.764

#Confidence interval
CI = sort(Y)        
CI[0.025*10000]      # -1560
CI[0.975*10000]      # 1500


######### 3 ############

#Frequency Table
set.seed(1)
freq_table = table(Y)

# Freq table:
#-----------------------------------------
#   Y     |     Observed     | Expected   |
#-----------------------------------------
#  1500   |       1535       | 0.1548     |
#-----------------------------------------
#  980    |       2151       | 0.2122824  |
#-----------------------------------------
#  460    |       1966       | 0.193812   |
#-----------------------------------------
#  -560   |       1499       | 0.1527376  |
#-----------------------------------------
# -1060   |       1963       | 0.195168   |
#-----------------------------------------
# -1560   |       886        | 0.0912     |
#-----------------------------------------

#Chi-squared goodness of fit test
#Null Hypothesis : There are no significant differences between Y distributions and X distributions
#Alternative Hypothesis : There are significant differences between Y distributions and X distributions
alpha = 0.05

#p = c(0.1548,0.2122824,0.193812,0.0912,0.195168,0.1527376)
#x = c(1500,980,980,980,460,460,460,460,-1560,-1060,-1060,-560,-560,-560)

ob = c(886,  1963,  1499,  1966,  2151,  1535 )
ex = c(0.1548,0.2122824,0.193812,0.0912,0.195168,0.1527376)

chisq.test(x = ob, p = ex)   #p-value < 2.2e-16 < 0.05, reject the null hypothesis.

##########################
