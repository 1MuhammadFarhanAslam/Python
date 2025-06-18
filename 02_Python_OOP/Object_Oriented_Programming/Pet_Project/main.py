# main.py
from pet import Pet # Pet class ko import kiya

# Chalo apna pehla Digital Pet banate hain!
my_dog = Pet("Buddy", "Dog")
my_cat = Pet("Whiskers", "Cat")
my_dragon = Pet("Smaug", "Dragon")


print("--- Game Start ---")

# Buddy ki shuruaati halat
my_dog.describe()

# Buddy ko khana khilate hain
my_dog.eat(30)
my_dog.describe()

# Buddy ko thoda khilate hain
my_dog.play(15)
my_dog.describe()

# Buddy ko mazeed khana khilate hain kyunki wo bhookha ho gaya
my_dog.eat(40)
my_dog.describe()

# Buddy ko sula dete hain
my_dog.sleep(8)
my_dog.describe()

# Ab Smaug (dragon) ki baari
my_dragon.describe()
my_dragon.play(50) # Dragon ko zyada khelna pasand hai
my_dragon.describe()
my_dragon.eat(10) # Thoda khana khilao
my_dragon.describe()
my_dragon.sleep(1) # Neend puri nahi hui!
my_dragon.describe()
my_dragon.sleep(7) # Ab puri neend le lo
my_dragon.describe()


print("\n--- Let's see how Whiskers is doing ---")
my_cat.describe()
my_cat.play(5) # Thoda sa khele
my_cat.eat(15) # Khana khaye
my_cat.describe()
my_cat.play(20) # Phir se khele, ab tired ho jaye ga
my_cat.describe() # Check energy and mood
my_cat.sleep(2) # Thodi neend le
my_cat.describe()
my_cat.play(10) # Phir se khele
my_cat.describe()


print("\n--- Game Over ---")