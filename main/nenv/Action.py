from main.nenv.Bid import Bid
from abc import ABC


class Action(ABC):
    """
        An agent should return a Negotiation Action during negotiation. An Action can be an Offer or an Accept. Also,
        each action obtains corresponding bid.
    """
    bid: Bid  #: Corresponding bid
    message: str  #: Optional message associated with the action

    def __init__(self, bid: Bid, message: str = None):
        self.bid = bid
        self.message = message


class Offer(Action):
    """
        If agent makes an offer (or counter-offer), it should return Offer Action.
    """
    def __str__(self):
        return "Offer: " + str(self.bid)

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return self.__str__().__hash__()

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()


class Accept(Offer):
    """
        If agent accepts the opponent's bid, it should return Accept Action.
    """
    def __str__(self):
        return "Accept: " + str(self.bid)

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return self.__str__().__hash__()

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()


class SubAgreement(Offer):
    """
        Accepts a sub-agreement on the opponent's last partial bid.
        The bid contains only the agreed-upon issues (partial bid).
        Used in the PARTIAL protocol to lock specific issues before full agreement.
    """
    def __str__(self):
        return "SubAgreement: " + str(self.bid)

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return self.__str__().__hash__()

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()


