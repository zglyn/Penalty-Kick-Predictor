Supervised role selector and cross-fitting
For each annotated frame, a click identifies the containing detection nearest its box center.
Ambiguous/missed matches are skipped. Other people become negative examples for that role.
Two CatBoost binary models learn role likelihoods from box geometry, pose and color features.
Click coordinates are supervision only, never inference features. Distinct role assignment is
chosen by model probabilities, with camera-aware continuity; there is no positional role prior.
Low-confidence/unavailable roles fall back to the full frame and explicit missingness indicators.
Training clips use source-cross-fitted role predictions; held-out clips use the complete allowed
training partition's selector. Inner selection repeats this entire procedure independently.
