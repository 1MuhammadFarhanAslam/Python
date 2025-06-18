# pet.py
from mood import Mood 

class Pet:
    """
    This class represents our digital pet, managing its core attributes and actions.
    It uses the Mood class for managing its emotional state.
    """
    def __init__(self, name: str, species: str):
        self.name = name
        self.species = species # E.g., 'Dog', 'Cat', 'Dragon'
        self.hunger_level = 50 # 0 (full) to 100 (starving)
        self.energy_level = 100 # 0 (exhausted) to 100 (full of energy)
        self.is_sleeping = False
        self.mood = Mood() # <<< Composition: Pet has Mood object!

    def eat(self, food_amount: int):
        """
        Makes the pet eat, reducing hunger and improving mood slightly.
        """
        if self.is_sleeping:
            print(f"{self.name} is sleeping! Cannot eat right now. 😴")
            return

        print(f"{self.name} is eating {food_amount} units of food.")
        self.hunger_level -= food_amount
        if self.hunger_level < 0:
            self.hunger_level = 0
        
        self.mood.change_mood(2) 
        print(f"{self.name}'s hunger is now {self.hunger_level}/100.")
        self._check_status() # Status check karo

    def play(self, play_time: int):
        """
        Makes the pet play, reducing energy and increasing hunger, affecting mood.
        """
        if self.is_sleeping:
            print(f"{self.name} is sleeping! Cannot play right now. 😴")
            return

        print(f"{self.name} is playing for {play_time} minutes!")
        self.energy_level -= (play_time * 2) # Playing consumes more energy
        self.hunger_level += play_time # Playing makes pet hungry

        if self.energy_level < 0:
            self.energy_level = 0
        if self.hunger_level > 100:
            self.hunger_level = 100

        # Mood based on energy and hunger
        if self.energy_level < 20 or self.hunger_level > 80:
            self.mood.change_mood(-3) # Zyada thakne ya bhookh se mood kharab
        else:
            self.mood.change_mood(3) # Normal khelne se mood acha

        print(f"{self.name}'s energy is now {self.energy_level}/100 and hunger is {self.hunger_level}/100.")
        self._check_status()

    def sleep(self, sleep_duration: int):
        """
        Makes the pet sleep, restoring energy and slightly reducing mood (due to being inactive).
        """
        if not self.is_sleeping:
            print(f"{self.name} is going to sleep for {sleep_duration} hours. 😴")
            self.is_sleeping = True
            # Mood halka sa kam hota hai neend se (kyunki abhi khel nahi raha)
            self.mood.change_mood(-1)
        else:
            print(f"{self.name} is already sleeping.")
            return # Agar pehle se so raha hai to mazeed kuch na karein

        # Neend ke baad energy restore karo
        self.energy_level += (sleep_duration * 10)
        if self.energy_level > 100:
            self.energy_level = 100
        
        print(f"{self.name} woke up! Energy is now {self.energy_level}/100.")
        self.is_sleeping = False # Neend khatam
        self._check_status()


    def _check_status(self):
        """
        Internal method to check pet's levels and suggest actions.
        """
        if self.hunger_level >= 80:
            print(f"Warning: {self.name} is very hungry! Consider feeding.")
            self.mood.change_mood(-2) # Bhookh se mood kharab

        if self.energy_level <= 20:
            print(f"Warning: {self.name} is very tired! Consider letting it sleep.")
            self.mood.change_mood(-2) # Thakawat se mood kharab

        print(f"Current Status - Hunger: {self.hunger_level}/100, Energy: {self.energy_level}/100, Mood: {self.mood.get_mood()}")


    def describe(self):
        """
        Provides a complete description of the pet's current state.
        """
        print(f"\n--- {self.name} ({self.species}) ---")
        print(f"  Hunger: {self.hunger_level}/100 {'(Hungry!)' if self.hunger_level >= 80 else ''}")
        print(f"  Energy: {self.energy_level}/100 {'(Tired!)' if self.energy_level <= 20 else ''}")
        print(f"  Mood: {self.mood.get_mood()}") # Composition ka istemal!
        print(f"  Sleeping: {'Yes' if self.is_sleeping else 'No'}")
        print("------------------------")