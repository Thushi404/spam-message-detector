import sys
from src.train import train_and_evaluate
from src.predict import predict_message, load_model




def display_prediction(result: dict) -> None:
    """
    Pretty-print the classification result returned by predict_message().

    Parameters
    ----------
    result : dict
        The dictionary returned by predict_message().
    """
    # Choose color/icon based on the prediction
    if result["prediction"] == "Spam":
        icon = "🚫"
        verdict = "SPAM"
        bar_color = "█" * int(result["confidence"] * 20)
    else:
        icon = "✅"
        verdict = "NOT SPAM"
        bar_color = "█" * int(result["confidence"] * 20)

    print()
    print("┌────────────────────── RESULT ──────────────────────┐")
    print(f"│  Prediction : {icon}  {verdict:<38}│")
    print(f"│  Confidence : {result['confidence'] * 100:.1f}%  "
          f"{bar_color:<32}│")
    print(f"│  Cleaned    : {result['cleaned_text'][:38]:<38}│")
    print("└────────────────────────────────────────────────────┘")
    print()


def run_prediction_mode() -> None:
    """
    Interactive prediction loop.

    Loads the model once, then lets the user classify as many messages
    as they want.  Type 'quit' or 'exit' to return to the main menu.
    """
    try:
        print("[INFO] Loading saved model...")
        model, vectorizer = load_model()
        print("[INFO] Model loaded successfully!\n")
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("[TIP]  Select option 1 from the main menu to train the model first.\n")
        return

    print("Enter an email or message to classify.")
    print("Type  'quit'  or  'exit'  to return to the main menu.\n")
    
    while True:
        try:
            user_input = input(" Enter message: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if user_input.lower() in ("quit", "exit", "q"):
            print("[INFO] Returning to main menu...\n")
            break

        if not user_input:
            print("[WARNING] Please enter a non-empty message.\n")
            continue

        try:
            result = predict_message(user_input, model=model, vectorizer=vectorizer)
            display_prediction(result)
        except Exception as e:
            print(f"[ERROR] Prediction failed: {e}\n")


def main() -> None:
    """
    Main loop — displays the menu and dispatches user choices.
    """
    print(BANNER)

    while True:
        print(MENU)
        try:
            choice = input("  ➤  Select an option (1/2/3): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[INFO] Goodbye!")
            sys.exit(0)

        if choice == "1":
            # ------ Train the model ------
            print("\n" + "=" * 55)
            print("        STARTING MODEL TRAINING")
            print("=" * 55 + "\n")
            try:
                train_and_evaluate()
            except Exception as e:
                print(f"\n[ERROR] Training failed: {e}")

        elif choice == "2":
            # ------ Predict mode ------
            run_prediction_mode()

        elif choice == "3":
            # ------ Exit ------
            print("\n[INFO] Thank you for using Spam Detection! Goodbye! \n")
            sys.exit(0)

        else:
            print("\n[WARNING] Invalid option. Please enter 1, 2, or 3.\n")


if __name__ == "__main__":
    main()
