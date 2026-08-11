# TODO

This table is not yet final, it will very likely change as the model improves. So be careful about using it right now.


`Rubin_objectId` : The Rubin DP1 objectId, which can be used to match this entry to other Rubin DP1 tables
`Gaia_sourceId` : The Gaia DR3 source_id, which can be used to match this entry to other Gaia DR3 tables.
`Y_pred` : The class predictions made by flexAE, as described in Géron et al. (2026). These are stored as floats between 0 (single star) and 1 (binary).
`is_binary` : A boolean flag that states whether this entry is a binary (True) or single star (False). This was determined by using Y_pred > 0.9 as a threshold. 