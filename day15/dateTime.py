import datetime
x = datetime.datetime.now()
# birthday date
y = datetime.datetime(2002, 8, 3)
print("Current date and time:", x)
# Print details of the specific date using strftime
print("Day of the week:", y.strftime("%A"))  # Full weekday name
print("Month name:", y.strftime("%B"))      # Full month name
print("Year:", y.strftime("%Y"))            # Full year
print("AM/PM:", y.strftime("%p"))           # AM/PM
print("Minutes:", y.strftime("%M"))         # Minutes (00-59)
print("Seconds:", y.strftime("%S"))         # Seconds (00-59)
print("Microseconds:", y.strftime("%f"))    # Microseconds (000000-999999)


