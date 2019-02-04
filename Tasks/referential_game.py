from Agents.receiver_agent import ReceiverAgent
from Agents.sender_agent import SenderAgent
import data_handler as dh


class Referential_Game:
    """
    Class to play referential game, where a sender sees a target image and must send a message to a receiver, who must
    the pick the target image from a set of candidate images.
    Attributes:
        Sender [Agent]
        Receiver [Agent]
        Vocabulary Size [Int]
        Distractor Set Size [Int]
    """

    def __init__(self, K=100, D=2):
        """

        :param sender: sender in game
        :param receiver: receiver in game
        :param K [Int]: Vocabulary Size
        :param D [Int]: Distractor Set Size
        """
        self.sender = SenderAgent()
        self.receiver = ReceiverAgent(self.sender.get_output(), self.sender.target_image)
        self.K = K # Vocabulary Size
        self.D = D # Distractor Set Size


    def play_game(self):
        """
        Play a single instance of the game
        :return: None
        """
        images = dh.get_images()
        target_image, distractors = images[0], images[1:]

        message, prediction, loss = self.receiver.run_game(target_image, distractors)