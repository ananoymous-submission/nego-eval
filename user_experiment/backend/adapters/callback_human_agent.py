"""Callback-based HumanAgent for UI-agnostic implementation.

Replaces queue-based GradioHumanAgent with cleaner callback pattern.
ALTERNATING-only: every offer must address all issues, accept only on a
complete bid.
"""

from typing import List, Optional, Callable
from main.HumanAgent.HumanAgent import HumanAgent
from main.nenv.Action import Action, Offer
from main.nenv.MessageType import MessageType
from main.nenv.Bid import Bid


class CallbackHumanAgent(HumanAgent):
    """Human agent that gets input via callback instead of terminal or queues."""

    def __init__(
        self,
        preference,
        session_time: int,
        estimators: List,
        input_callback: Callable[[], str],
        output_callback: Optional[Callable[[str], None]] = None,
        name: str = "Human",
        model_name: Optional[str] = None,
    ):
        super().__init__(preference, session_time, estimators, name, model_name)
        self.input_callback = input_callback
        self.output_callback = output_callback

    def initiate(self, opponent_name: Optional[str]):
        pass

    def receive_action(self, action, t: float, chat_history: List[str] = None):
        super().receive_action(action, t, chat_history)
        if self.output_callback and chat_history and len(chat_history) > 0:
            self.output_callback(chat_history[-1])

    def act(self, t: float, chat_history: List[str] = None) -> Action:
        """Get human's decision via callback. Validates that every offer is
        complete (all issues present) and that accepts only happen on a
        complete last received bid."""
        while True:
            result = self.input_callback()
            if isinstance(result, tuple):
                if len(result) >= 3:
                    message, bid_dict, locked_dict = result[0], result[1], result[2]
                else:
                    message, bid_dict = result
                    locked_dict = {}
            else:
                message, bid_dict, locked_dict = result, None, {}

            classification = self.classifier.classify(message, chat_history)
            msg_type = classification['type']

            if msg_type == MessageType.ACCEPT:
                try:
                    last_bid = self.last_received_bids[-1]
                    bid_issues = set(issue.name for issue in last_bid.content.keys())
                    all_issues = set(issue.name for issue in self.preference.issues)
                    if bid_issues == all_issues:
                        return self.accept_action
                    error_msg = self._format_error_message(
                        "CANNOT_ACCEPT_INCOMPLETE",
                        missing_issues=all_issues - bid_issues,
                    )
                    self._send_error_and_retry(error_msg)
                    continue
                except (IndexError, AttributeError):
                    self._send_error_and_retry(self._format_error_message("NO_OFFER_TO_ACCEPT"))
                    continue

            if msg_type == MessageType.OFFER:
                issue_map = {issue.name: issue for issue in self.preference.issues}
                bid_dict = bid_dict or {}
                locked_dict = locked_dict or {}
                bid_content = {
                    issue_map[name]: value
                    for name, value in bid_dict.items()
                    if name in issue_map
                }
                has_unlocked_selection = any(
                    name not in locked_dict or str(locked_dict[name]) != str(value)
                    for name, value in bid_dict.items()
                    if name in issue_map
                )

                if not bid_content or not has_unlocked_selection:
                    self._send_error_and_retry(self._format_error_message("PANEL_REQUIRED"))
                    continue

                bid = Bid(bid_content)
                bid_issues = set(issue.name for issue in bid.content.keys())
                all_issues = set(issue.name for issue in self.preference.issues)
                missing_issues = all_issues - bid_issues
                if missing_issues:
                    self._send_error_and_retry(
                        self._format_error_message("INCOMPLETE_OFFER", missing_issues=missing_issues)
                    )
                    continue

                bid.is_complete = True
                bid.utility = self.preference.get_utility(bid)
                return Offer(bid=bid, message=message)

            self._send_error_and_retry(self._format_error_message("NOT_AN_OFFER"))

    def _send_error_and_retry(self, error_message: str):
        if self.output_callback:
            self.output_callback(error_message)

    def _format_error_message(self, error_type: str, missing_issues: set = None) -> str:
        all_issue_names = [issue.name for issue in self.preference.issues]

        if error_type == "PANEL_REQUIRED":
            return (
                "❌ **Use The Offer Panel** / **Teklif Panelini Kullanın**\n\n"
                f"Offers must be submitted through the offer panel. "
                f"Please choose values there for all issues: {', '.join(all_issue_names)}\n\n"
                f"Teklifler teklif paneli üzerinden gönderilmelidir. "
                f"Lütfen panelden tüm konular için değer seçin: {', '.join(all_issue_names)}"
            )

        if error_type == "INCOMPLETE_OFFER":
            missing = ', '.join(sorted(missing_issues))
            return (
                "❌ **Incomplete Offer** / **Eksik Teklif**\n\n"
                f"Your offer must include ALL negotiation issues. Missing: {missing}\n\n"
                f"Teklifiniz TÜM müzakere konularını içermelidir. Eksik: {missing}\n\n"
                f"Required issues / Gerekli konular: {', '.join(all_issue_names)}"
            )

        if error_type == "CANNOT_ACCEPT_INCOMPLETE":
            missing = ', '.join(sorted(missing_issues))
            return (
                "❌ **Cannot Accept Incomplete Offer** / **Eksik Teklif Kabul Edilemez**\n\n"
                f"The last offer is incomplete (missing: {missing}). You cannot accept it.\n\n"
                f"Son teklif eksik (eksik: {missing}). Eksik bir teklifi kabul edemezsiniz."
            )

        if error_type == "NO_OFFER_TO_ACCEPT":
            return (
                "❌ **No Offer to Accept** / **Kabul Edilecek Teklif Yok**\n\n"
                "You haven't received an offer yet. Wait for your opponent's offer first.\n\n"
                "Henüz bir teklif almadınız. Önce rakibinizin teklifini bekleyin."
            )

        if error_type == "NOT_AN_OFFER":
            return (
                "❌ **Invalid Message** / **Geçersiz Mesaj**\n\n"
                "**English:** Not a valid message. Your message can be an offer or an acceptance.\n\n"
                "**Türkçe:** Geçersiz mesaj. Mesajınız bir teklif veya kabul olmalıdır."
            )

        return (
            "❌ **Invalid Input** / **Geçersiz Girdi**\n\n"
            "Please provide a valid negotiation message.\n\n"
            "Lütfen geçerli bir müzakere mesajı sağlayın."
        )
