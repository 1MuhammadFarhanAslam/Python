class Mood():
    def __init__(self):
        # Mood level: -10 (very sad) to +10 (very happy)
        self._mood_level = 0 # _ shows it is internal variable
        self._update_status()

    def _update_status(self):
        """
        Internal method to update the mood status based on _mood_level.
        """
        if self._mood_level > 5:
            self.status = "Very Happy! 😊"
        elif self._mood_level > 0:
            self.status = "Happy 😄"
        elif self._mood_level == 0:
            self.status = "Neutral 😐"
        elif self._mood_level < -5:
            self.status = "Very Sad! 😭"
        else:
            self.status = "A bit down 😔"

    def get_mood(self):
        """
        Returns the current mood status.
        """
        return self.status

    def change_mood(self, amount: int):
        """
        Changes the mood level by a given amount.
        Positive amount makes pet happier, negative makes it sadder.
        Mood level stays between -10 and +10.
        """
        self._mood_level += amount
        # Mood level ko bounds ke andar rakho
        if self._mood_level > 10:
            self._mood_level = 10
        elif self._mood_level < -10:
            self._mood_level = -10
        self._update_status() # Mood change hone ke baad status update karo
        print(f"Mood changed by {amount}. Current mood: {self.get_mood()}")

    def __str__(self):
        """
        Special method to provide a string representation of the object.
        Used when you print the object directly.
        """
        return f"Current mood: {self.get_mood()} (Level: {self._mood_level})"