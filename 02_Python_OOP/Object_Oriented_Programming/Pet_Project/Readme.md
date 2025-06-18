**logic building** (the way you think about solving problems). Don't worry, we'll break down how I built the logic for this project step-by-step. Before starting any project, no matter how small, laying down a solid foundation is essential.

Think of it like building a house. You don't just start laying bricks immediately. First, you think about the design and create a blueprint. Programming is quite similar.

---

### How to Build Project Logic (Simplified Approach)

Whenever you want to build a project, big or small, follow these steps:

#### Step 1: Understand the Idea (Brainstorming)

* **What's the project?** A Digital Pet.
* **What will it do?** It will be a virtual animal.
* **What can the animal do?** It can eat, play, and sleep.
* **What characteristics will the animal have?** A name, species (like dog/cat), hunger level, energy level (tiredness), and a mood (happy/sad).

---

#### Step 2: Identify Core "Objects" (Find the Real-World Things)

OOP is all about real-world objects. So, what are the main "things" we're working with in this project?

* The primary thing is the **Pet** itself. This will be our central object.
* What else is inside the pet? Its hunger, energy – these are just numbers.
* But "mood" feels like a separate concept. Mood isn't just happy or sad; it has levels. Does the pet *is a* mood, or does the pet *has a* mood? The pet *has a* mood. When you think "has a," that's where you consider **Composition**. So, a separate **Mood** object (class) could be useful.

**So, we've identified 2 core objects (classes):**
1.  `Pet`
2.  `Mood`

---

#### Step 3: Define Each Object's Characteristics (Properties/Attributes) and Actions (Methods)

Now, let's think about each class separately:

#### **For the `Mood` Class:**

* **Characteristics (Data - Variables):**
    * What's the mood's numeric level? (e.g., -10 to +10). Let's call it `_mood_level`. The underscore `_` hints that this is an internal variable and shouldn't be directly changed from outside.
    * Based on `_mood_level`, what's the current `status`? (e.g., "Happy", "Sad"). Let's call this `status`.
* **Actions (Methods):**
    * How is the mood set initially? The `__init__` method.
    * How do we change the mood? `change_mood(amount)`.
    * How do we get the current mood state? `get_mood()`.
    * How do we update the `status` based on `_mood_level`? `_update_status()`. This method will be called inside `change_mood`.
    * What should happen when we `print()` a Mood object directly? The `__str__` method.

---

#### **For the `Pet` Class:**

* **Characteristics (Data - Variables):**
    * What's its name? `name`
    * What species is it? `species`
    * How hungry is it? `hunger_level` (0-100)
    * How much energy does it have? `energy_level` (0-100)
    * Is it sleeping or not? `is_sleeping` (True/False)
    * **And most importantly:** What's its mood? This is where we'll place the `Mood` class object! `self.mood = Mood()`. This is **Composition**!
* **Actions (Methods):**
    * How do we set the name and species when creating a new pet? The `__init__` method.
    * How do we feed the pet? `eat(food_amount)`.
        * Eating should decrease hunger.
        * How does it affect mood? It will call `mood.change_mood()`.
        * How does it affect energy? Probably not much directly.
    * How do we make the pet play? `play(play_time)`.
        * Playing should decrease energy.
        * It should increase hunger.
        * How does it affect mood? If it gets too tired/hungry, its mood might worsen; otherwise, it should improve. It will call `mood.change_mood()`.
    * How do we make the pet sleep? `sleep(duration)`.
        * Sleeping should increase energy.
        * How does it affect mood? Maybe a slight decrease (because it's inactive). It will call `mood.change_mood()`.
        * It should update the `is_sleeping` status.
    * How do we check the pet's overall condition? `describe()`.
        * This will print `hunger_level`, `energy_level`, `is_sleeping`, and critically, **`mood.get_mood()`** (notice, we're calling a method on the `Mood` object inside the `Pet` class!).
    * How do we check for critical conditions (like being too hungry or tired)? Create an internal helper method, `_check_status()`, which will be called after `eat`, `play`, or `sleep`.

---

#### Step 4: Create a Flow Chart or Pseudocode (Rough Sketch)

Mentally, or on paper, outline a rough flow:

* **Start**
* **Create Pet:**
    * Provide Name, Species
    * Initialize Hunger (50), Energy (100), Sleeping (False)
    * **Create Mood Object (`Mood()`) and assign it to `self.mood`**
* **Actions Loop (e.g., in `main.py`):**
    * Call `pet.describe()` to see initial state.
    * Call `pet.eat()`.
    * Call `pet.play()`.
    * Call `pet.sleep()`.
    * Call `pet.describe()` again to see changes.
* **End**

I've simplified the flow here; you could make it more interactive with user input.

---

#### Step 5: Write the Code and Test It (Trial and Error)

Now, translate everything you've thought about into actual code.
* First, write the `Mood` class. Test each method to ensure it works correctly.
* Then, write the `Pet` class. When you need to use the `Mood` class object, remember to put `self.mood = Mood()` in the `__init__` method.
* Implement the logic for each method (`eat`, `play`, `sleep`).
* In `main.py`, create `Pet` objects and call their methods to test them. Use `print` statements to see if the values are changing as expected.

**You WILL make mistakes!**
* You'll get `TypeError`s (missing parameters, wrong types).
* You'll get `AttributeError`s (calling a variable name incorrectly).
* The logic might not work as intended (hunger increases instead of decreasing).

This is completely normal! Make mistakes, learn from them. That's what programming is all about. Google your errors, debug your code (run it line by line to see what's happening).

---

#### How Composition Was Used in This Project:

* The `Pet` class does **not** inherit from the `Mood` class (meaning a `Pet` is **not a** `Mood`).
* Instead, the `Pet` class **has an** object of the `Mood` class (`self.mood = Mood()`).
* When the `Pet` needs to change its mood, it calls methods on its `self.mood` object (e.g., `self.mood.change_mood()`).
* This approach keeps the `Mood` class solely responsible for managing mood, and the `Pet` class solely responsible for managing the pet's core behavior. Both classes perform their own tasks but work together to form a larger system.

This is the process of building logic. Break things into smaller pieces, think about each piece individually, and then put them back together. It might seem challenging at first, but it gets easier with practice.