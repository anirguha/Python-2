class Letter:
    """ A Letter indicates a single English letter from a guess, and
        has methods and attributes to describe whether the letter was in
        (or in the correct place) in the hidden target word.
    """

    def __init__(self, letter: str) -> None:
        self.letter = letter
        self.in_correct_place: bool = False
        self.in_word: bool = False

    def is_in_correct_place(self) -> bool:
        return self.in_correct_place

    def is_in_word(self) -> bool:
        return self.in_word

class DisplaySpecification:
    """A dataclass for holding display specifications for WordyPy. The following
    values are defined:
        - block_width: the width of each character in pixels
        - block_height: the height of each character in pixels
        - correct_location_color: the hex code to color the block when it is correct
        - incorrect_location_color: the hex code to color the block when it is in the wrong location but exists in the string
        - incorrect_color: the hex code to color the block when it is not in the string
        - space_between_letters: the amount of padding to put between characters, in pixels
        - word_color: the hex code of the background color of the string
    """

    block_width: int = 80
    block_height: int = 80
    correct_location_color: str = "#00274C"
    incorrect_location_color: str = "#FFCB05"
    incorrect_color: str = "#D3D3D3"
    space_between_letters: int = 5
    word_color: str = "#FFFFFF"


from PIL import Image


class Bot:
    """
    Bot class is to define the player
    """
    def _tuple_to_str(self, pixel: tuple) -> str:
        r, g, b = pixel[:3]
        return f"{r:02X}{g:02X}{b:02X}"

    # favorite_words = ["doggy", "drive", "daddy", "field", "state"]

    def __init__(self, word_list_file: str, display_spec: DisplaySpecification) -> None:
        with open(word_list_file, 'r') as file:
            self.word_list: list[str] = list(set([line.strip() for line in file if line.strip()]))
        self.previous_guesses = []  # List to track previous guesses
        self.correct_letters = [None] * 5  # To store the correct letters at each position (initially unknown)
        self.misplaced_letters = set()  # To store letters that are in the word but misplaced
        self.incorrect_letters = set()  # To store letters that are confirmed not in the word
        self.display_spec = display_spec  # To store the display specifications

    # def _tuple_to_str(self, pixel: tuple) -> str:
    #     r, g, b = pixel[:3]
    #     return f"{r:02X}{g:02X}{b:02X}"

    def _hex_to_rgb(self, hex_color: str) -> tuple[int, ...]:
        """Helper function to convert hex color like '#FFCB05' to (255, 203, 5)"""
        hex_color = hex_color.lstrip("#")
        return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))

    def _process_image(self, guess: str, guess_image: Image.Image) -> list["Letter"]:
        """Process the guess feedback image and create a list of Letter objects."""

        # Convert the display spec hex codes into RGB tuples
        correct_color = self._hex_to_rgb(self.display_spec.correct_location_color)
        incorrect_location_color = self._hex_to_rgb(self.display_spec.incorrect_location_color)
        incorrect_color = self._hex_to_rgb(self.display_spec.incorrect_color)

        block_width = self.display_spec.block_width
        block_height = self.display_spec.block_height
        space_between_letters = self.display_spec.space_between_letters

        results = []

        for idx, letter in enumerate(guess):
            # Calculate (x, y) sampling point
            x = idx * (block_width + space_between_letters) + block_width // 2
            y = block_height // 2

            pixel_color = guess_image.getpixel((x, y))

            # If the pixel has 4 channels (RGBA), drop alpha
            if isinstance(pixel_color, tuple) and len(pixel_color) == 4:
                pixel_color = pixel_color[:3]

            ltr = Letter(letter)

            if pixel_color == correct_color:
                ltr.in_correct_place = True
            elif pixel_color == incorrect_location_color:
                ltr.in_word = True
            else:
                pass  # wrong letter, leave both attributes False

            results.append(ltr)

        return results

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

    def record_guess_results(self, guess: str, guess_results: Image) -> None:
        """
        Process the results of a guess and adjust internal state.
        - Track the correct positions based on exact mattches.
        - Track letters that are in the word but in wrong positon.
        """
        self.guess_results = guess_results
        self.guess = guess
        self.guess_results = self._process_image(guess, guess_results)

from PIL import Image, ImageFont, ImageDraw
import random
import os

def display(img):
    """Display the image using PIL's show method"""
    img.show()

class GameEngine:
    """The GameEngine represents a new WordPy game to play."""

    def __init__(self, display_spec: DisplaySpecification = None) -> None:
        """Creates a new WordyPy game engine. If the game_spec is None then
        the engine will use the default color and drawing values, otherwise
        it will override the defaults using the provided specification
        """
        # det the display specification to defaults or user provided values
        if display_spec == None:
            display_spec = DisplaySpecification()
        self.display_spec = display_spec

        self.err_input = False
        self.err_guess = False
        self.prev_guesses = []  # record the previous guesses

    def play(
        self, bot: Bot, word_list_file: str = "words.txt", target_word: str = None
    ) -> Image:
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

        # read in the dictionary of allowable words
        word_list: list[str] = list(
            map(lambda x: x.strip().upper(), open(word_list_file, "r").readlines())
        )
        # record the known correct positions
        known_letters: list[str] = [None] * 5
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
            correct, results = self._set_feedback(guess, target_word)

            # print out a line indicating whether the guess was correct or not
            print(f"Was this guess correct? {correct}")

            # get the image to be returned to the caller
            img = self._format_results(results)

            print(f"Sending guess results to bot:\n")
            display(img)

            bot.record_guess_results(guess, img)

            # if they got it correct we can just end
            if correct:
                print(f"Great job, you found the target word in {i} guesses!")
                return

        # if we get here, the bot didn't guess the word
        print(
            f"Thanks for playing! You didn't find the target word in the number of guesses allowed."
        )
        return

    def _set_feedback(self, guess: str, target_word: str) -> tuple[bool, list[Letter]]:
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
            else:
                # we know they don't have a perfect answer, so let's update
                # our correct variable for feedback
                correct = False

            # check to see if this character is anywhere in the word
            if guess[j] in target_word:
                letter.in_word = True

            # add this letter to our list of letters
            letters.append(letter)

        return correct, letters

    def _render_letter(self, letter: Letter) -> Image:
        """This function renders a single Letter object as an image."""
        # set color string as appropriate
        color: str = self.display_spec.incorrect_color
        if letter.is_in_correct_place():
            color = self.display_spec.correct_location_color
        elif letter.is_in_word():
            color = self.display_spec.incorrect_location_color

        # now we create a new image of width x height with the given color
        block = Image.new(
            "RGB",
            (self.display_spec.block_width, self.display_spec.block_height),
            color=color,
        )
        # and we actually render that image and get a handle back
        draw = ImageDraw.Draw(block)

        # for the lettering we need to identify the center of the block,
        # so we calculate that as the (X,Y) position to render text
        X: int = self.display_spec.block_width // 2
        Y: int = self.display_spec.block_height // 2

        # we will create a font object for drawing lettering
        FONT_SIZE: int = 50
        try:
            # Try loading Arial font which is commonly available
            font = ImageFont.truetype("Arial", FONT_SIZE)
        except OSError:
            # If Arial is not available, use the default font
            font = ImageFont.load_default()

        # now we can draw the letter and tell PIL we want to have the
        # character centered in the box using the anchor attribute
        draw.text((X, Y), letter.letter, font=font, anchor="mm")

        return block

    def _format_results(self, letters: list[Letter]) -> Image:
        """This function does the hard work of converting the list[Letter]
        for a guess into an image.
        """
        # some constants that determine what a word of these letters
        # will look like. The algorithm for rendering a word is that
        # we will render each letter independently and put some spacing between
        # them. This means the total word width is equal to the size of
        # all of the letters and the spacing, and the word height is equal
        # to the size of just a single letter
        WORD_WIDTH: int = (len(letters) * self.display_spec.block_width) + (
            len(letters) - 1
        ) * self.display_spec.space_between_letters
        WORD_HEIGHT: int = self.display_spec.block_height

        # we can use the paste() function to place one PIL.Image on top
        # of another PIL.Image
        word = Image.new(
            "RGB", (WORD_WIDTH, WORD_HEIGHT), color=self.display_spec.word_color
        )
        curr_loc = 0
        for letter in letters:
            # we can render the letter and then paste, setting the location
            # as X,Y position we want to paste it in
            rendered_letter: Image = self._render_letter(letter)
            word.paste(rendered_letter, (curr_loc, 0))
            curr_loc += (
                self.display_spec.block_width + self.display_spec.space_between_letters
            )

        return word

if __name__ == "__main__":
        # Chris's favorite words
        favorite_words = ["doggy", "drive", "daddy", "field", "state"]

        # Write this to a temporary file
        words_file = "temp_file.txt"
        with open(words_file, "w") as file:
            file.writelines("\n".join(favorite_words))

        # Create a new GameEngine with the default DisplaySpecification
        ge = GameEngine()

        # Initialize the student Bot using the display specification from the game engine object
        bot = Bot(words_file, ge.display_spec)

        # Play a game with the Bot
        ge.play(bot, word_list_file=words_file)