Label-free extraction and camera model
Cache detector candidates, poses, timestamps, sampled RGB frames, and background transforms.
Click timestamps select an additional exact annotation-nearest frame for role training only.
They do not alter the classification frame sampling. All videos use the same sampling rule.
A similarity transform maps previous-frame pixels to current-frame pixels. It models
translation, in-plane rotation and uniform zoom. Person masks exclude foreground motion.
RANSAC rejects outliers; invalid fits carry quality=0 and are never treated as valid compensation.
