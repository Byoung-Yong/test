from __future__ import annotations

import run_hidden_family_dataset_validation as validation

# Current official GitHub blob SHA for Imperssonator/OFET-Database:
# Functions/OFETDatabase.mat. The previous value referred to an older blob.
validation.P3HT_GIT_BLOB_SHA1 = "5ce52b1fb505f3bb2c5341a57349381713195b60"

if __name__ == "__main__":
    validation.main()
