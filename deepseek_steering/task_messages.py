task_messages = [
    # Mathematical Logic
    {"role": "user", "content": "What is the smallest number that when divided by 2,3,4,5 and 6 always leaves a remainder of 1?"},
    {"role": "user", "content": "What is the sum of all two-digit numbers that are both perfect squares and multiples of 3?"},
    {"role": "user", "content": "What's the largest three-digit palindrome number that's divisible by 11?"},
    
    # Spatial Reasoning
    {"role": "user", "content": "If you stack cubes in a pyramid where each layer is a square and the top layer is 1x1, and you use exactly 30 cubes total, how many layers does the pyramid have?"},
    {"role": "user", "content": "In a 3x3 grid of points, how many different squares can you make by connecting the points (squares of any size, but must be square)?"},
    {"role": "user", "content": "If you fold a rectangular piece of paper in half lengthwise, then in half widthwise, then cut off one corner, how many holes will appear when you unfold it?"},
    
    # Verbal Logic
    {"role": "user", "content": "Complete this analogy: Odometer is to distance as clock is to ___"},
    {"role": "user", "content": "Complete: Caterpillar is to butterfly as tadpole is to ___"},
    {"role": "user", "content": "Complete: Second is to minute as minute is to ___"},
    
    # Pattern Recognition
    {"role": "user", "content": "What comes next in the sequence: 2,6,12,20,30,___"},
    {"role": "user", "content": "What number comes next: 1,1,2,3,5,8,13,___"},
    {"role": "user", "content": "What comes next in the sequence: 1,4,9,16,25,___"},
    
    # Lateral Thinking
    {"role": "user", "content": "What 5-letter word becomes shorter when you add two letters to it?"},
    {"role": "user", "content": "What has a head and a tail but no body?"},
    {"role": "user", "content": "What word in English is always spelled incorrectly?"},
    
    # Causal Reasoning
    {"role": "user", "content": "If all A are B, and all B are C, and this is not C, what can you conclude about it being A?"},
    {"role": "user", "content": "If all squares are rectangles, and this shape is not a rectangle, can it be a square? Answer yes or no."},
    {"role": "user", "content": "If all mammals are animals, and all dogs are mammals, is every animal a dog? Answer yes or no."},
    
    # Probabilistic Thinking
    {"role": "user", "content": "If you roll a fair six-sided die twice, what's the probability (as a percentage rounded to the nearest whole number) of getting a sum of 7?"},
    {"role": "user", "content": "In a standard deck of 52 cards, what's the probability (as a percentage rounded to the nearest whole number) of drawing a red ace?"},
    {"role": "user", "content": "If you flip three fair coins, what's the probability (as a percentage rounded to the nearest whole number) of getting exactly two heads?"},
    
    # Constraint Reasoning
    {"role": "user", "content": "How many different ways can you arrange the letters in 'BOOK'?"},
    {"role": "user", "content": "In how many ways can you arrange 3 identical red balls and 2 identical blue balls in a row?"},
    {"role": "user", "content": "How many different 4-digit numbers can be formed using the digits 1,1,2,2?"},
    
    # Creative Problem Solving
    {"role": "user", "content": "How many triangles are there in a regular hexagon (including all possible triangles formed by its vertices)?"},
    {"role": "user", "content": "If you have 9 coins and need to identify one fake coin that weighs less than the others using a balance scale, what is the minimum number of weighings needed to guarantee finding it?"},
    {"role": "user", "content": "What's the minimum number of straight cuts needed to divide a circular cake into 8 equal pieces?"},
    
    # Scientific Reasoning
    {"role": "user", "content": "If a pendulum's length is doubled, by what factor does its period (time for one complete swing) increase? Express as a decimal with 1 decimal place."},
    {"role": "user", "content": "If you mix 100ml of water at 20°C with 100ml of water at 40°C in an insulated container, what will be the final temperature in Celsius?"},
    {"role": "user", "content": "A ball is thrown straight up with an initial velocity of 20 m/s. How many seconds will it take to reach its maximum height? (Use g=10 m/s² and ignore air resistance)"},
]

task_answers = [
    # Mathematical Logic
    "61",  # Smallest number with remainder 1 when divided by 2,3,4,5,6
    "117",  # Sum of two-digit square multiples of 3 (36 + 81)
    "979",  # Largest three-digit palindrome divisible by 11
    
    # Spatial Reasoning
    "4",  # Layers in pyramid of 30 cubes (1+4+9+16=30)
    "14",  # Number of squares in 3x3 grid
    "4",   # Number of holes after folding and cutting
    
    # Verbal Logic
    "time",  # Odometer:distance :: clock:time
    "frog",  # Caterpillar:butterfly :: tadpole:frog
    "hour",  # Second:minute :: minute:hour
    
    # Pattern Recognition
    "42",  # Next in sequence (difference increases by 2)
    "21",  # Fibonacci sequence
    "36",  # Square numbers
    
    # Lateral Thinking
    "short",  # Becomes "shorter"
    "coin",  # Head and tail riddle
    "incorrectly",  # Word riddle
    
    # Causal Reasoning
    "not A",  # Logic conclusion
    "no",  # Square/rectangle logic
    "no",  # Mammals/animals logic
    
    # Probabilistic Thinking
    "17",  # Die probability (6/36 * 100)
    "4",  # Red ace probability (2/52 * 100)
    "38",  # Three coins probability
    
    # Constraint Reasoning
    "12",  # BOOK arrangements (4!/(2!))
    "10",  # Arrangements of red and blue balls
    "6",   # Numbers from 1,1,2,2
    
    # Creative Problem Solving
    "20",  # Triangles in regular hexagon
    "2",   # Minimum weighings for fake coin
    "3",   # Cuts for 8 pieces
    
    # Scientific Reasoning
    "1.4",  # Pendulum period increases by √2 ≈ 1.4 (period ∝ √length)
    "30",   # Temperature averaging due to conservation of energy
    "2",    # Time to max height = initial velocity / acceleration due to gravity
]
