# Versioning

Given AITK contains every recipe and is versioned, we could safely version our recipes and do not care backward compatiblity.

So there will be two cases:

## Version mismatch

We will notify user that there is new version of recipe and user could:

- Downgrade to previous version to use the recipes
- Replace everything automatically for the user
- Manually merge the recipes and update the version

### When developer should increase the version

Developers don't need to increase the version until release, they should do it whenever they changes needed files in pr.
(So in AITK release, version could jump from 1 to 7 like so)

Should increase:

- Update Olive json for new features
- Update user script for new features
- Update inference sample for functional features like how to setup ort etc.

Maynot need:

- Update README.md
- Update *.json.config
- Update inference sample for non functional features like metrics, new samples etc.

## Version match

When versions are same, we could patch automatically for new files for user.

### Incremental patching

If the main version is not changed, we should support incremental patching, i.e. when new configs are added or new features are added, the previous created projects should support it smoothly.

This requires that if the new template will modify existing user script, ipynb etc., it should not do so and create new one instead.
