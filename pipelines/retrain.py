import os
import sys
from datetime import datetime

def retrain():
    print("=" * 50)
    print(" Retraining Pipeline Started")
    print(" Time:", datetime.utcnow())
    print("=" * 50)

    # Step 1: Train model
    exit_code = os.system("python src/train.py")

    if exit_code != 0:
        print(" Training failed")
        sys.exit(1)

    print(" Training completed successfully")

    
    print(" Restart API manually if needed")
    print(" New model ready for deployment")

    print("=" * 50)
    print(" Retraining Pipeline Finished")
    print("=" * 50)


if __name__ == "__main__":
    retrain()
