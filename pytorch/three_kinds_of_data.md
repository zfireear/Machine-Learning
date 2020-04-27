# three kinds of data

## Interval, ordinal, and categorical values  
You should be aware of three kinds of numerical  values as you attempt to makesense of your data. 

The first kind is continuous values. These values are the most intuitive when represented as numbers; they’re strictly ordered, and a difference between various values has a strict meaning. 

> Stating that package A is 2 kilograms heavier than package B or that package B came from 100 miles farther away than package A has a fixed meaning, no matter whether package A weighs 3 kilograms or 10, or whether B came from 200 miles away or 2,000. 
  
If you’re counting or measuring something with units,the value probably is a continuous value.

Next are ordinal values. The strict ordering of continuous values remains, but the fixed relationship between values no longer applies.  
> A good example is ordering a small, medium, or large drink, with small mapped to the value 1, medium to 2, and large to 3. The large drink is bigger than the medium, in the same way that 3 is bigger than 2, but it doesn’t tell you anything about how much bigger. If you were to convert 1, 2, and 3 to the actual volumes (say, 8, 12, and 24 fluid ounces), those values would switch to interval values. **It’s important to remember that you can’t do math on the values beyond ordering them**; trying to average large=3 and small=1 does not result in a medium drink!

Finally, categorical values have neither ordering nor numerical meaning. These values are often enumerations of possibilities, assigned arbitrary numbers. 
> Assigning water to 1, coffee to 2, soda to 3, and milk to 4 is a good example. Placing water first and milk last has no real logic; you simply need distinct values to differentiate them. You could assign coffee to 10 and milk to –3 with no significant change (although assigning valuesin the range 0..N-1 will have advantages when we discuss one-hot encoding later).

