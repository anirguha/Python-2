class Letter:
    """ A Letter indicates a single English letter from a guess, and
        has methods and attributes to describe whether the letter was in
        (or in the correct place) in the hidden target word.
    """

    def __init__(self, letter: str) -> None:
        self.letter: str = letter
        self.in_correct_place: bool = False
        self.in_word: bool = False

    def is_in_correct_place(self) -> bool:
        return  self.in_correct_place

    def is_in_word(self) -> bool:
        return self.in_word


import random

class Bot:
    """
    Bot class is to define the player
    """
    # favorite_words = ["doggy", "drive", "daddy", "field", "state"]

    def __init__(self, word_list_file: str) -> None:
        with open(word_list_file, 'r') as file:
            self.word_list: list[str] = list(set([line.strip() for line in file if line.strip()]))
        self.previous_guesses = []  # List to track previous guesses
        self.correct_letters = [None] * 5  # To store the correct letters at each position (initially unknown)
        self.misplaced_letters = set() # To store letters that are in the word but misplaced
        self.incorrect_letters = set()  # To store letters that are confirmed not in the word

    def make_guess(self) -> str:
        """
        Makes a guess based on available information and the word list.
        Adjusts future guesses based on previous feedback.
        """
        if not self.previous_guesses:
            # If it's the first guess, choose randomly from the word list
            self.previous_guesses.append(random.choice(self.word_list).upper())
            return self.previous_guesses[-1]

        # Filter out words that are incompatible with the known feedback
        possible_words = []
        for word in self.word_list:
            if self.is_valid_guess(word.upper()):
                possible_words.append(word.upper())

        return random.choice(possible_words).upper() if possible_words else random.choice(self.word_list).upper()

    def is_valid_guess(self, word: str) -> bool:
        """
        Check if a word is a valid guess based on previous guesses and feedback.
        - Avoid words that contradict the correct positions of letters.
        - Avoid words containing letters that are not in the word.
        """
        for i, ch in enumerate(word):
            if self.correct_letters[i] is not None and word[i] != self.correct_letters[i]:
                return False

            # 2) Must include all yellows somewhere
        if any(c not in word for c in self.misplaced_letters):
            return False

            # 3) Must not include any grays
        if any(c in self.incorrect_letters for c in word):
            return False

            # 4) No repeat
        if word in self.previous_guesses:
            return False

        return True

    def record_guess_results(self, guess: str, results: list) -> None:
        """
                Process the results of a guess and adjust internal state.
                - Update correct positions based on exact matches.
                - Track letters that are in the word but in the wrong position.
        """
        for i, res in enumerate(results):
            letter = res.letter
            if res.in_correct_place:
                # green
                self.correct_letters[i] = letter
            elif res.in_word:
                # yellow
                self.misplaced_letters.add(letter)
            else:
                # gray
                self.incorrect_letters.add(letter)

import random


class GameEngine:
    """The GameEngine represents a new WordPy game to play."""

    def __init__(self):
        self.err_input = False
        self.err_guess = False
        self.prev_guesses = []  # record the previous guesses

    def play(
        self, bot, word_list_file: str = "words.txt", target_word: str = None
    ) -> None:
        """Plays a new game, using the supplied bot. By default the GameEngine
        will look in words.txt for the list of allowable words and choose one
        at random. Set the value of target_word to override this behavior and
        choose the word that must be guessed by the bot.
        """

        def format_results(results) -> str:
            """Small function to format the results into a string for quick
            review by caller.
            """
            response = ""
            for letter in results:
                if letter.is_in_correct_place():
                    response = response + letter.letter
                elif letter.is_in_word():
                    response = response + "*"
                else:
                    response = response + "?"
            return response

        def set_feedback(guess: str, target_word: str) -> tuple[bool, list[Letter]]:
            # whether the complete guess is correct
            # set it to True initially and then switch it to False if any letter doesn't match
            correct: bool = True

            letters = []
            for j in range(len(guess)):
                # create a new Letter object
                letter = Letter(guess[j])

                # check to see if this character is in the same position in the
                # guess and if so set the in_correct_place attribute
                if guess[j] == target_word[j]:
                    letter.in_correct_place = True
                    known_letters[j] = guess[j]  # record the known correct positions
                else:
                    # we know they don't have a perfect answer, so let's update
                    # our correct variable for feedback
                    correct = False

                # check to see if this character is anywhere in the word
                if guess[j] in target_word:
                    letter.in_word = True
                else:
                    unused_letters.add(guess[j])  # record the unused letters

                # add this letter to our list of letters
                letters.append(letter)

            return correct, letters

        # read in the dictionary of allowable words
        word_list: list[str] = list(
            map(lambda x: x.strip().upper(), open(word_list_file, "r").readlines())
        )
        # record the known correct positions
        known_letters: list[str] = [None, None, None, None, None]
        # set of unused letters
        unused_letters = set()

        # assign the target word to a member variable for use later
        if target_word is None:
            target_word = random.choice(word_list).upper()
        else:
            target_word = target_word.upper()
            if target_word not in word_list:
                print(f"Target word {target_word} must be from the word list")
                self.err_input = True
                return

        print(
            f"Playing a game of WordyPy using the word list file of {word_list_file}.\nThe target word for this round is {target_word}\n"
        )

        MAX_GUESSES = 6
        for i in range(1, MAX_GUESSES):
            # ask the bot for it's guess and evaluate
            guess: str = bot.make_guess()

            # print out a line indicating what the guess was
            print(f"Evaluating bot guess of {guess}")

            if guess not in word_list:
                print(f"Guessed word {guess} must be from the word list")
                self.err_guess = True
            elif guess in self.prev_guesses:
                print(f"Guess word cannot be the same one as previously used!")
                self.err_guess = True

            if self.err_guess:
                return

            self.prev_guesses.append(guess)  # record the previous guess

            for j, letter in enumerate(guess):
                if letter in unused_letters:
                    print(
                        f"The bot's guess used {letter} which was previously identified as not used!"
                    )
                    self.err_guess = True
                if known_letters[j] is not None:
                    if letter != known_letters[j]:
                        print(
                            f"Previously identified {known_letters[j]} in the correct position is not used at position {j}!"
                        )
                        self.err_guess = True

                if self.err_guess:
                    return

            # get the results of the guess
            correct, results = set_feedback(guess, target_word)

            # print out a line indicating whether the guess was correct or not
            print(f"Was this guess correct? {correct}")

            print(f"Sending guess results to bot {format_results(results)}\n")

            bot.record_guess_results(guess, results)

            # if they got it correct we can just end
            if correct:
                print(f"Great job, you found the target word in {i} guesses!")
                return

        # if we get here, the bot didn't guess the word
        print(
            f"Thanks for playing! You didn't find the target word in the number of guesses allowed."
        )
        return

if __name__ == "__main__":
    # Chris's favorite words
    # favorite_words = ["doggy", "drive", "daddy", "field", "state"]

    # Write this to a temporary file
    words_file = "words.txt"
    # with open(words_file, "w") as file:
    #     file.writelines("\n".join(favorite_words))

    # Initialize the student Bot
    bot = Bot(words_file)

    # Create a new GameEngine and play a game with the Bot, in this
    # tests run I chose to set the target_word to "doggy"
    GameEngine().play(bot, word_list_file=words_file, target_word="table")