install.packages("dplyr")
install.packages("ggplot2")
install.packages("descr")
install.packages("psych")
library(dplyr)
library(ggplot2)
library(descr)
library(psych)

#importing data & a few basic checks
data = read.csv(file.choose(), header = T, na.strings=c("","NA"), strip.white = TRUE)
data
View(data)      #Viewing the data
nrow(data)    #total rows
ncol(data)    #total columns
names(data)   #variable names
data.frame(sapply(data, class))   #columns data types
describe(data)                    #basic statistical details about data
str(data)       #structure of the data
head(data)

#checkiing NA and NULL values 
any(is.null(data))             #checking for any null values in data
any(is.na(data))               #checking for any NA values in data
summary(data)   #summary of the data

#Imputing mean values in the NA columns
data$children[is.na(data$children)]<-mean(data$children,na.rm=TRUE)
any(is.na(data))      

#Exploring the data

#Number of booking at both the hotels
table(data$hotel)                                  #tabular form
ggplot(data = data, aes(x = hotel)) +              #Graphical form
  geom_bar(stat = "count") +
  labs(title = "Number of booking at both the hotels",
       x = "Hotel type",
       y = "No. of bookings")

# Hotel types and the market segments for each hotel type
ggplot(data = data) +
  geom_bar(mapping = aes(x = hotel, fill = market_segment)) +
  labs(title = "Mostly Chosen Hotel type", subtitle = "Based on Market Segment")

#distribution of hotel type for cancellation
table(data$is_canceled, data$hotel)                #tabular form

ggplot(data = data,                                #Barplot
       aes(
         x = hotel,
         y = prop.table(stat(count)),
         fill = factor(is_canceled),
         label = scales::percent(prop.table(stat(count)))
       )) +
  geom_bar(position = position_dodge()) +
  geom_text(
    stat = "count",
    position = position_dodge(.9),
    vjust = -0.5,
    size = 3
  ) +
  scale_y_continuous(labels = scales::percent) +
  labs(title = "Cancellation Status by Hotel Type",
       x = "Hotel Type",
       y = "Count") +
  theme_classic() +
  scale_fill_discrete(
    name = "Booking Status",
    breaks = c("0", "1"),
    labels = c("Cancelled", "Not Cancelled")
  )

############################
ggplot(data = data, aes(                     #Boxplot
  x = hotel,
  y = lead_time,
  fill = factor(is_canceled)
)) +
  geom_boxplot(position = position_dodge()) +
  labs(
    title = "Cancellation By Hotel Type",
    subtitle = "Based on Lead Time",
    x = "Hotel Type",
    y = "Lead Time (Days)"
  ) +
  scale_fill_discrete(
    name = "Booking Status",
    breaks = c("0", "1"),
    labels = c("Cancelled", "Not Cancelled")
  ) + theme_light()


#####################################
unique(data$arrival_date_year)        #Exploring Arrival date year

# Organizing data based on Months
data$arrival_date_month =                                
  factor(data$arrival_date_month, levels = month.name)

# Visualizing data Monthly basis
ggplot(data = data, aes(x = arrival_date_month)) +    
  geom_bar(fill = "skyblue") +
  geom_text(stat = "count", aes(label = ..count..), hjust = 1) +
  coord_flip() + labs(title = "Booking requests on Monthly basis",
                      x = "Month", y = "Count") + theme_classic()

# Booking status on a Monthly basis
ggplot(data, aes(arrival_date_month, fill = factor(is_canceled))) +
  geom_bar() + geom_text(stat = "count", aes(label = ..count..), hjust = 1) +
  coord_flip() + scale_fill_discrete(
    name = "Booking Status",
    breaks = c("0", "1"),
    label = c("Cancelled", "Not Cancelled")) +
  labs(title = "Booking Status by Month",
       x = "Month", y = "Count") + theme_bw()

#Booking confirmation and not the checkins
ggplot(data, aes(arrival_date_month, fill = hotel)) +
  geom_bar(position = position_dodge()) +
  labs(title = "Booking Status by Month",
       x = "Month", y = "Count") + theme_bw()

####################################################
data_booking = data[data$reservation_status == "Check-Out",]

#Subsetting the data based on the places people made the bookings
sub_booking = data_booking %>% 
  group_by(country) %>% 
  filter(n() > 2000)

# Country wise bookings
install.packages("countrycode")
library(countrycode)

sub_booking$county_name = countrycode(sub_booking$country, 
                                     origin = "iso3c",
                                     destination = "country.name")

# Traveller by Country per hotel wise
ggplot(sub_booking, aes(county_name, fill = hotel)) + 
  geom_bar(stat = "count", position = position_dodge()) + 
  labs(title = "Booking Status by Country",
       x = "Country",
       y = "Count") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1),
        panel.background = element_blank())

#Booking by country
library(tidyverse)
data %>% summarise (Portugal = round(mean(country == "PRT"), 3) * 100,
                          Other_countries = round(mean(country != "PRT"), 3) * 100) %>%
  pivot_longer(
    cols = c("Portugal",  "Other_countries"),
    names_to = "Region",
    values_to = "value"
  ) %>%
  mutate(Region = str_replace_all(Region, "_", " "),
         lab.ypos = cumsum(value) - 0.5 * value) %>%
  arrange(value) %>% 
  
  ggplot(aes(
    x = 2,
    y = value,
    fill = Region,
    label = value
  )) +
  geom_bar(stat = "identity", color = "white") +
  geom_text(aes(y = lab.ypos), color = "white") +
  coord_polar(theta = "y", start = 0) +
  theme_void() +
  xlim(0.5, 2.5) +
  theme() +
  ggtitle("Booking Rate by country of origin") +
  scale_fill_hue(c = 50, l = 40)


###########################################
# Duration of total stay
ggplot(sub_booking, aes(stays_in_weekend_nights + stays_in_week_nights)) + 
  geom_density(col = "blue") +facet_wrap(~hotel) + theme_bw()

# Average daily prices by each of the Hotels
ggplot(sub_booking, aes(x = adr, fill = hotel, color = hotel)) + 
  geom_histogram(aes(y = ..density..), position = position_dodge(), binwidth = 20 ) +
  geom_density(alpha = 0.2) + 
  labs(title = "Average daily prices by each of the Hotels",
       x = "Hotel Price(in Euro)",
       y = "Count") + scale_color_brewer(palette = "Paired") + 
  theme_classic() + theme(legend.position = "top")

############################################

#Hotel preferences by customer types
ggplot(sub_booking, aes(customer_type, fill = hotel)) + 
  geom_bar(stat = "count", position = position_dodge()) + 
  labs(title = "Hotel Preference by Customer Type",
       x = "Customer Type",
       y = "Count") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1),
        panel.background = element_blank())

# Does the hotel charged differently for different customer type
ggplot(sub_booking, aes(x = customer_type, y = adr, fill = hotel)) + 
  geom_boxplot(position = position_dodge()) + 
  labs(title = "Price Charged by Hotel Type",
       subtitle = "for Customer Type",
       x = "Customer Type",
       y = "Price per night(in Euro)") + theme_classic()

############################################
#Analyzing Distribution Channel

ggplot(data = data) +
  geom_bar(mapping = aes(x = distribution_channel)) +
  labs(title = "Distribution Channel Comparison", subtitle = "Number of transaction in each channel")

# Segment the bar chart group by "deposit_type"
ggplot(data = data) +
  geom_bar(mapping = aes(x = distribution_channel, fill = deposit_type)) +
  labs(title = "Distribution ChannelComparison", subtitle = "Segmented by Deposit Type")

ggplot(data = data) +
  geom_bar(mapping = aes(x = distribution_channel, fill = market_segment)) +
  labs(title = "Distribution Channel Comparison", subtitle = "Group By Market Segment")

############################################
#Analyzing Family based data

# Correlation between booking's lead time and guests who bring their children
ggplot(data = data) +
  geom_point(mapping = aes(x = lead_time, y = children)) +
  labs(title = "Any Correlation Between Lead Time and Guest's Children?", subtitle = "Lead Time Versus Number of Children Bringing By The Guest") +
  annotate("text", x = 300, y = 3.5, label = "Shorter lead times for guests with fewer children", color = "Blue", size =4.5)


#########################################
