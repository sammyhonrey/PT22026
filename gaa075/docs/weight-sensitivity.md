# Checkpoint C3 — feature-weight sensitivity analysis

Read-only analysis over the approved Phase 4 matrix. No recommender is
implemented; this document exists to gather evidence before Phase 5.

## Method

- Catalogue: **89,583 tracks**, matrix **123 columns** (9 numeric + 114 genre), built by `build_feature_matrix()`.
- `S_num` = cosine similarity of the 9 standardised, row-normalised audio features.
- `S_genre` = cosine similarity of the 114 multi-label genre indicators.
- Combined score = `alpha * S_num + beta * S_genre`, with `beta = 1 - alpha`.
- Top 10 per reference, reference track itself excluded, ties broken by `track_id`.
- Selected reference tracks are deterministic (most popular in each genre profile, ties by `track_id`).
- Sanity check: `cosine(weighted matrix)` vs `0.7*S_num + 0.3*S_genre` differ by at most **4.44e-16** — the block decomposition holds.

**Interpretation guard:** every number below is a cosine similarity between feature
vectors. It is not a percentage, not a probability, and not a prediction that a
listener will like a track.

## Reference tracks

| id | reference | artist | genres | popularity |
|---|---|---|---|---|
| 7DbdUf8aHSYoliSjO6LZv6 | Sex, Drugs, Etc. | Beach Weather | rock | 90 |
| 2tTmW7RDtMQtBk7m2rYeSw | Quevedo: Bzrp Music Sessions, Vol. 52 | Bizarrap;Quevedo | hip-hop | 99 |
| 5NGtFXVpXSvwunEIGeviY3 | Gymnopédie No. 1 | Erik Satie;Philippe Entremont | classical | 70 |
| 4ih3dyFZoeTdaeJW9mPbOI | Mr. Vain - Original Radio Edit | Culture Beat | techno | 68 |
| 5ww2BF9slyYgNOk37BlC4u | La Bachata | Manuel Turizo | latin, latino, reggae, reggaeton | 98 |
| 6S3JlDAGk3uu3NtZbPnuhS | Baby Blue - Remastered 2010 | Badfinger | blues, country, folk, j-pop +5 | 67 |

## Duplicate songs that de-duplication by `track_id` does not catch

The Phase 4 catalogue is one row per `track_id`, and the 24,259 repeated-id rows are
gone. But **the same recording can carry two different `track_id`s**, and those rows
survive de-duplication:

- distinct `(track_name, artists)` pairs in the catalogue: **81,198**
- pairs with more than one `track_id`: **4,647**
- rows involved: **13,032** (14.5% of the catalogue)
- largest group: **45** rows for the same title and artist

Example found directly in a rock top-10: *Follow You* by Imagine Dragons appears twice
with different ids (`1skWrgJrEi6ZB4Wt5tYYAb`, `7FdUvDkaE24o3FPIWTvzv2`), identical tempo
and energy, popularity 2 vs 73. This is why a top-10 can contain the same song twice —
a real defect against the brief's "avoid returning duplicate recommendations".

| group type | groups |
|---|---|
| same title+artist, identical audio features (same recording) | **2,331** |
| same title+artist, differing audio features (possibly a different version) | **2,316** |

So a duplicate-suppression rule at ranking time must be deliberate: collapsing every
same-title/artist group would also hide legitimate live/remaster variants whose audio
features differ. This needs an explicit decision in Phase 5, not a silent fix.

## Top-10 at the extremes and at the current default

### single-genre rock (popular) — *Sex, Drugs, Etc.* (rock)

**alpha=0.50 / beta=0.50** — 10/10 share at least one genre; 9/10 distinct songs; mean S_num 0.830, mean S_genre 1.000

| # | track | artist | genres | combined | S_num | S_genre |
|---|---|---|---|---|---|---|
| 1 | Follow You | Imagine Dragons | rock | 0.970 | 0.939 | 1.000 |
| 2 | Follow You | Imagine Dragons | rock | 0.970 | 0.939 | 1.000 |
| 3 | Fly Away | Lenny Kravitz | rock | 0.967 | 0.935 | 1.000 |
| 4 | Radioactive | Imagine Dragons | rock | 0.938 | 0.875 | 1.000 |
| 5 | Don't Stop Me Now - Remastered 2011 | Queen | rock | 0.937 | 0.874 | 1.000 |
| 6 | World's Smallest Violin | AJR | rock | 0.885 | 0.770 | 1.000 |
| 7 | Are You Gonna Go My Way | Lenny Kravitz | rock | 0.884 | 0.769 | 1.000 |
| 8 | human - Passion Pit Remix | Christina Perri;Passion Pit | rock | 0.879 | 0.758 | 1.000 |
| 9 | Everyday I Love You Less And Less | Kaiser Chiefs | rock | 0.862 | 0.725 | 1.000 |
| 10 | Infinity | Jaymes Young | rock | 0.858 | 0.715 | 1.000 |

**alpha=0.70 / beta=0.30** — 10/10 share at least one genre; 8/10 distinct songs; mean S_num 0.900, mean S_genre 0.899

| # | track | artist | genres | combined | S_num | S_genre |
|---|---|---|---|---|---|---|
| 1 | Follow You | Imagine Dragons | rock | 0.958 | 0.939 | 1.000 |
| 2 | Follow You | Imagine Dragons | rock | 0.958 | 0.939 | 1.000 |
| 3 | Fly Away | Lenny Kravitz | rock | 0.954 | 0.935 | 1.000 |
| 4 | Radioactive | Imagine Dragons | rock | 0.913 | 0.875 | 1.000 |
| 5 | Don't Stop Me Now - Remastered 2011 | Queen | rock | 0.912 | 0.874 | 1.000 |
| 6 | My Songs Know What You Did In The Dark (Light Em Up) | Fall Out Boy | emo, rock | 0.893 | 0.973 | 0.707 |
| 7 | My Songs Know What You Did In The Dark (Light Em Up) | Fall Out Boy | emo, rock | 0.893 | 0.973 | 0.707 |
| 8 | In Over My Head | grandson | alt-rock, alternative, rock | 0.841 | 0.955 | 0.577 |
| 9 | World's Smallest Violin | AJR | rock | 0.839 | 0.770 | 1.000 |
| 10 | Are You Gonna Go My Way | Lenny Kravitz | rock | 0.838 | 0.769 | 1.000 |

**alpha=0.90 / beta=0.10** — 9/10 share at least one genre; 8/10 distinct songs; mean S_num 0.952, mean S_genre 0.652

| # | track | artist | genres | combined | S_num | S_genre |
|---|---|---|---|---|---|---|
| 1 | My Songs Know What You Did In The Dark (Light Em Up) | Fall Out Boy | emo, rock | 0.946 | 0.973 | 0.707 |
| 2 | My Songs Know What You Did In The Dark (Light Em Up) | Fall Out Boy | emo, rock | 0.946 | 0.973 | 0.707 |
| 3 | Follow You | Imagine Dragons | rock | 0.945 | 0.939 | 1.000 |
| 4 | Follow You | Imagine Dragons | rock | 0.945 | 0.939 | 1.000 |
| 5 | Fly Away | Lenny Kravitz | rock | 0.941 | 0.935 | 1.000 |
| 6 | In Over My Head | grandson | alt-rock, alternative, rock | 0.917 | 0.955 | 0.577 |
| 7 | The Last DJ | Tom Petty and the Heartbreakers | folk, rock, singer-songwriter +1 | 0.897 | 0.942 | 0.500 |
| 8 | Fish Rock | Thaikkudam Bridge | indian, indie, indie-pop +2 | 0.896 | 0.946 | 0.447 |
| 9 | Distrito Federal - Vivo en el Palacio de los Deportes | Los Auténticos Decadentes | ska | 0.894 | 0.993 | 0.000 |
| 10 | Adam's Song | blink-182 | punk, punk-rock, rock | 0.890 | 0.925 | 0.577 |

### single-genre hip-hop (popular) — *Quevedo: Bzrp Music Sessions, Vol. 52* (hip-hop)

**alpha=0.50 / beta=0.50** — 10/10 share at least one genre; 10/10 distinct songs; mean S_num 0.815, mean S_genre 1.000

| # | track | artist | genres | combined | S_num | S_genre |
|---|---|---|---|---|---|---|
| 1 | Sadi Gali | Lehmber Hussainpuri | hip-hop | 0.944 | 0.888 | 1.000 |
| 2 | Superman | Eminem;Dina Rae | hip-hop | 0.927 | 0.854 | 1.000 |
| 3 | Once Upon a Time | Anirudh Ravichander | hip-hop | 0.919 | 0.838 | 1.000 |
| 4 | Oh Ho Ho Ho (Remix) | Sukhbir;Ikka | hip-hop | 0.911 | 0.822 | 1.000 |
| 5 | Tabahi | Badshah | hip-hop | 0.907 | 0.815 | 1.000 |
| 6 | Ethir Neechal | Anirudh Ravichander;Yo Yo Honey Singh;Hiphop Tamizha | hip-hop | 0.897 | 0.795 | 1.000 |
| 7 | Jaat Da Muqabala | Sidhu Moose Wala | hip-hop | 0.896 | 0.792 | 1.000 |
| 8 | Vilambara Idaiveli - From "Imaikkaa Nodigal" | Hiphop Tamizha;Christopher Stanley;Sudarshan Ashok;Srinisha Jayaseelan | hip-hop | 0.893 | 0.786 | 1.000 |
| 9 | Calaboose | Sidhu Moose Wala | hip-hop | 0.893 | 0.785 | 1.000 |
| 10 | Coldplay | Lizzo | hip-hop | 0.888 | 0.777 | 1.000 |

**alpha=0.70 / beta=0.30** — 10/10 share at least one genre; 10/10 distinct songs; mean S_num 0.831, mean S_genre 0.971

| # | track | artist | genres | combined | S_num | S_genre |
|---|---|---|---|---|---|---|
| 1 | Sadi Gali | Lehmber Hussainpuri | hip-hop | 0.922 | 0.888 | 1.000 |
| 2 | Superman | Eminem;Dina Rae | hip-hop | 0.898 | 0.854 | 1.000 |
| 3 | Once Upon a Time | Anirudh Ravichander | hip-hop | 0.887 | 0.838 | 1.000 |
| 4 | Oh Ho Ho Ho (Remix) | Sukhbir;Ikka | hip-hop | 0.875 | 0.822 | 1.000 |
| 5 | Tabahi | Badshah | hip-hop | 0.870 | 0.815 | 1.000 |
| 6 | Sirikkalam Parakkalam | Benny Dayal;Madurai Soulijos | hip-hop, pop-film | 0.869 | 0.938 | 0.707 |
| 7 | Ethir Neechal | Anirudh Ravichander;Yo Yo Honey Singh;Hiphop Tamizha | hip-hop | 0.856 | 0.795 | 1.000 |
| 8 | Jaat Da Muqabala | Sidhu Moose Wala | hip-hop | 0.854 | 0.792 | 1.000 |
| 9 | Vilambara Idaiveli - From "Imaikkaa Nodigal" | Hiphop Tamizha;Christopher Stanley;Sudarshan Ashok;Srinisha Jayaseelan | hip-hop | 0.851 | 0.786 | 1.000 |
| 10 | Calaboose | Sidhu Moose Wala | hip-hop | 0.850 | 0.785 | 1.000 |

**alpha=0.90 / beta=0.10** — 2/10 share at least one genre; 9/10 distinct songs; mean S_num 0.963, mean S_genre 0.171

| # | track | artist | genres | combined | S_num | S_genre |
|---|---|---|---|---|---|---|
| 1 | Sirikkalam Parakkalam | Benny Dayal;Madurai Soulijos | hip-hop, pop-film | 0.915 | 0.938 | 0.707 |
| 2 | Sadi Gali | Lehmber Hussainpuri | hip-hop | 0.900 | 0.888 | 1.000 |
| 3 | Human Touch | Armin van Buuren;Sam Gray | progressive-house, trance | 0.886 | 0.984 | 0.000 |
| 4 | Human Touch | Armin van Buuren;Sam Gray | progressive-house, trance | 0.884 | 0.982 | 0.000 |
| 5 | Sangre para el Rey | OConnor | heavy-metal | 0.879 | 0.977 | 0.000 |
| 6 | 魔界舞曲 | Seikima-II | heavy-metal | 0.876 | 0.974 | 0.000 |
| 7 | COLOR ME UP! | Flower | j-idol | 0.874 | 0.972 | 0.000 |
| 8 | Sometimes | DubVision;The Him | progressive-house | 0.874 | 0.971 | 0.000 |
| 9 | I Really Like You | Carly Rae Jepsen | electro | 0.873 | 0.970 | 0.000 |
| 10 | Verdammt lang her - Best Of VRR | Andreas Gabalier | party | 0.873 | 0.970 | 0.000 |

### single-genre classical — *Gymnopédie No. 1* (classical)

**alpha=0.50 / beta=0.50** — 10/10 share at least one genre; 9/10 distinct songs; mean S_num 0.986, mean S_genre 1.000

| # | track | artist | genres | combined | S_num | S_genre |
|---|---|---|---|---|---|---|
| 1 | 3 Gymnopédies: No. 1 Lent et douloureux | Erik Satie;Alice Sara Ott | classical | 0.997 | 0.994 | 1.000 |
| 2 | 3 Gymnopédies: I. Lent et douloureux | Erik Satie;Anne Queffélec | classical | 0.995 | 0.991 | 1.000 |
| 3 | Piano Sonata No. 17 in B-Flat Major, K. 570: II. Adagio | Wolfgang Amadeus Mozart;Giuseppe Greco | classical | 0.994 | 0.988 | 1.000 |
| 4 | Préludes / Book 1, L.117: 8. La fille aux cheveux de lin | Claude Debussy;Jacques Février | classical | 0.993 | 0.985 | 1.000 |
| 5 | Piano Sonata No. 4 in E-Flat Major, K. 282: 1a. Adagio - Pt. 1 | Wolfgang Amadeus Mozart;Claire-Marie Le Guay | classical | 0.992 | 0.984 | 1.000 |
| 6 | Les petits riens, K.app.10 (ballet): No. 11 Larghetto | Wolfgang Amadeus Mozart;Concerto Köln;Anton Steck | classical | 0.992 | 0.984 | 1.000 |
| 7 | Revelation: Music in Pure Intonation: Night Vigil II | Michael Harrison | classical | 0.992 | 0.984 | 1.000 |
| 8 | Hourglass | Jiří Horák | classical | 0.992 | 0.984 | 1.000 |
| 9 | Préludes / Book 1, L. 117: 8. La fille aux cheveux de lin | Claude Debussy;Jean-Rodolphe Kars | classical | 0.991 | 0.981 | 1.000 |
| 10 | Préludes / Book 1, L. 117: 8. La fille aux cheveux de lin | Claude Debussy;Jean-Rodolphe Kars | classical | 0.991 | 0.981 | 1.000 |

**alpha=0.70 / beta=0.30** — 10/10 share at least one genre; 9/10 distinct songs; mean S_num 0.986, mean S_genre 1.000

| # | track | artist | genres | combined | S_num | S_genre |
|---|---|---|---|---|---|---|
| 1 | 3 Gymnopédies: No. 1 Lent et douloureux | Erik Satie;Alice Sara Ott | classical | 0.996 | 0.994 | 1.000 |
| 2 | 3 Gymnopédies: I. Lent et douloureux | Erik Satie;Anne Queffélec | classical | 0.994 | 0.991 | 1.000 |
| 3 | Piano Sonata No. 17 in B-Flat Major, K. 570: II. Adagio | Wolfgang Amadeus Mozart;Giuseppe Greco | classical | 0.992 | 0.988 | 1.000 |
| 4 | Préludes / Book 1, L.117: 8. La fille aux cheveux de lin | Claude Debussy;Jacques Février | classical | 0.990 | 0.985 | 1.000 |
| 5 | Piano Sonata No. 4 in E-Flat Major, K. 282: 1a. Adagio - Pt. 1 | Wolfgang Amadeus Mozart;Claire-Marie Le Guay | classical | 0.989 | 0.984 | 1.000 |
| 6 | Les petits riens, K.app.10 (ballet): No. 11 Larghetto | Wolfgang Amadeus Mozart;Concerto Köln;Anton Steck | classical | 0.989 | 0.984 | 1.000 |
| 7 | Revelation: Music in Pure Intonation: Night Vigil II | Michael Harrison | classical | 0.989 | 0.984 | 1.000 |
| 8 | Hourglass | Jiří Horák | classical | 0.989 | 0.984 | 1.000 |
| 9 | Préludes / Book 1, L. 117: 8. La fille aux cheveux de lin | Claude Debussy;Jean-Rodolphe Kars | classical | 0.987 | 0.981 | 1.000 |
| 10 | Préludes / Book 1, L. 117: 8. La fille aux cheveux de lin | Claude Debussy;Jean-Rodolphe Kars | classical | 0.987 | 0.981 | 1.000 |

**alpha=0.90 / beta=0.10** — 10/10 share at least one genre; 9/10 distinct songs; mean S_num 0.986, mean S_genre 1.000

| # | track | artist | genres | combined | S_num | S_genre |
|---|---|---|---|---|---|---|
| 1 | 3 Gymnopédies: No. 1 Lent et douloureux | Erik Satie;Alice Sara Ott | classical | 0.994 | 0.994 | 1.000 |
| 2 | 3 Gymnopédies: I. Lent et douloureux | Erik Satie;Anne Queffélec | classical | 0.992 | 0.991 | 1.000 |
| 3 | Piano Sonata No. 17 in B-Flat Major, K. 570: II. Adagio | Wolfgang Amadeus Mozart;Giuseppe Greco | classical | 0.989 | 0.988 | 1.000 |
| 4 | Préludes / Book 1, L.117: 8. La fille aux cheveux de lin | Claude Debussy;Jacques Février | classical | 0.987 | 0.985 | 1.000 |
| 5 | Piano Sonata No. 4 in E-Flat Major, K. 282: 1a. Adagio - Pt. 1 | Wolfgang Amadeus Mozart;Claire-Marie Le Guay | classical | 0.986 | 0.984 | 1.000 |
| 6 | Les petits riens, K.app.10 (ballet): No. 11 Larghetto | Wolfgang Amadeus Mozart;Concerto Köln;Anton Steck | classical | 0.986 | 0.984 | 1.000 |
| 7 | Revelation: Music in Pure Intonation: Night Vigil II | Michael Harrison | classical | 0.986 | 0.984 | 1.000 |
| 8 | Hourglass | Jiří Horák | classical | 0.985 | 0.984 | 1.000 |
| 9 | Préludes / Book 1, L. 117: 8. La fille aux cheveux de lin | Claude Debussy;Jean-Rodolphe Kars | classical | 0.983 | 0.981 | 1.000 |
| 10 | Préludes / Book 1, L. 117: 8. La fille aux cheveux de lin | Claude Debussy;Jean-Rodolphe Kars | classical | 0.983 | 0.981 | 1.000 |

### single-genre techno — *Mr. Vain - Original Radio Edit* (techno)

**alpha=0.50 / beta=0.50** — 10/10 share at least one genre; 10/10 distinct songs; mean S_num 0.846, mean S_genre 1.000

| # | track | artist | genres | combined | S_num | S_genre |
|---|---|---|---|---|---|---|
| 1 | Burn Dem Bridges | Skin On Skin | techno | 0.959 | 0.919 | 1.000 |
| 2 | Ready Or Not - Radio Edit | Cascada | techno | 0.948 | 0.896 | 1.000 |
| 3 | Dear Jessie | Rollergirl | techno | 0.933 | 0.867 | 1.000 |
| 4 | See The Light | Neelix;Omiki | techno | 0.932 | 0.864 | 1.000 |
| 5 | Run Away | Real McCoy | techno | 0.914 | 0.829 | 1.000 |
| 6 | Stay - Jonas Blue Remix | Zedd;Alessia Cara;Jonas Blue | techno | 0.914 | 0.827 | 1.000 |
| 7 | Spectrum | Zedd;Matthew Koma | techno | 0.911 | 0.822 | 1.000 |
| 8 | Inhale | Duke Dumont;Ebenezer | techno | 0.907 | 0.815 | 1.000 |
| 9 | Captain Jack - Short Mix | Captain Jack | techno | 0.906 | 0.811 | 1.000 |
| 10 | Get A Way | Maxx | techno | 0.905 | 0.811 | 1.000 |

**alpha=0.70 / beta=0.30** — 10/10 share at least one genre; 10/10 distinct songs; mean S_num 0.859, mean S_genre 0.971

| # | track | artist | genres | combined | S_num | S_genre |
|---|---|---|---|---|---|---|
| 1 | Burn Dem Bridges | Skin On Skin | techno | 0.943 | 0.919 | 1.000 |
| 2 | Ready Or Not - Radio Edit | Cascada | techno | 0.927 | 0.896 | 1.000 |
| 3 | Dear Jessie | Rollergirl | techno | 0.907 | 0.867 | 1.000 |
| 4 | See The Light | Neelix;Omiki | techno | 0.905 | 0.864 | 1.000 |
| 5 | Run Away | Real McCoy | techno | 0.880 | 0.829 | 1.000 |
| 6 | Stay - Jonas Blue Remix | Zedd;Alessia Cara;Jonas Blue | techno | 0.879 | 0.827 | 1.000 |
| 7 | Spectrum | Zedd;Matthew Koma | techno | 0.876 | 0.822 | 1.000 |
| 8 | It's My Rave | Locked Club | industrial, techno | 0.871 | 0.941 | 0.707 |
| 9 | Inhale | Duke Dumont;Ebenezer | techno | 0.870 | 0.815 | 1.000 |
| 10 | Captain Jack - Short Mix | Captain Jack | techno | 0.868 | 0.811 | 1.000 |

**alpha=0.90 / beta=0.10** — 7/10 share at least one genre; 10/10 distinct songs; mean S_num 0.918, mean S_genre 0.591

| # | track | artist | genres | combined | S_num | S_genre |
|---|---|---|---|---|---|---|
| 1 | Burn Dem Bridges | Skin On Skin | techno | 0.927 | 0.919 | 1.000 |
| 2 | It's My Rave | Locked Club | industrial, techno | 0.917 | 0.941 | 0.707 |
| 3 | Ready Or Not - Radio Edit | Cascada | techno | 0.906 | 0.896 | 1.000 |
| 4 | Dear Jessie | Rollergirl | techno | 0.880 | 0.867 | 1.000 |
| 5 | See The Light | Neelix;Omiki | techno | 0.877 | 0.864 | 1.000 |
| 6 | Packard - Nora En Pure Remix | Oliver Schories;Jan Blomqvist;Nora En Pure | minimal-techno, techno | 0.875 | 0.894 | 0.707 |
| 7 | 熱力節拍Wou Bom Ba | Grasshopper;Shirley Kwan;湯寶如;Winnie Lau | cantopop | 0.875 | 0.973 | 0.000 |
| 8 | It's My Life | Dr. Alban | house | 0.865 | 0.961 | 0.000 |
| 9 | Insomnia 2021 - Epic Edit | Maceo Plex;Faithless | deep-house, disco, minimal-techno +1 | 0.864 | 0.905 | 0.500 |
| 10 | Da Industry | DJ Sneak;IMIURU | chicago-house | 0.861 | 0.957 | 0.000 |

### multi-genre (>=4 genres) — *La Bachata* (latin, latino, reggae, reggaeton)

**alpha=0.50 / beta=0.50** — 10/10 share at least one genre; 7/10 distinct songs; mean S_num 0.820, mean S_genre 1.000

| # | track | artist | genres | combined | S_num | S_genre |
|---|---|---|---|---|---|---|
| 1 | Todo Bien | Lalo Ebratt;Juanes;Skinny Happy;Yera;Trapical | latin, latino, reggae +1 | 0.940 | 0.880 | 1.000 |
| 2 | Ram Pam Pam | Natti Natasha;Becky G | latin, latino, reggae +1 | 0.930 | 0.861 | 1.000 |
| 3 | Quizas - Remix | Tony Dize;Ken-Y;Marioso | latin, latino, reggae +1 | 0.922 | 0.843 | 1.000 |
| 4 | Quizas - Remix | Tony Dize;Ken-Y;Marioso | latin, latino, reggae +1 | 0.922 | 0.843 | 1.000 |
| 5 | La Copa | Ozuna | latin, latino, reggae +1 | 0.921 | 0.842 | 1.000 |
| 6 | Happy Happy | Nacho;Los Mendoza | latin, latino, reggae +1 | 0.901 | 0.803 | 1.000 |
| 7 | Gasolina | Daddy Yankee | latin, latino, reggae +1 | 0.894 | 0.788 | 1.000 |
| 8 | Secreto | Anuel AA;KAROL G | latin, latino, reggae +1 | 0.890 | 0.780 | 1.000 |
| 9 | Secreto | Anuel AA;KAROL G | latin, latino, reggae +1 | 0.890 | 0.780 | 1.000 |
| 10 | Secreto | Anuel AA;KAROL G | latin, latino, reggae +1 | 0.890 | 0.780 | 1.000 |

**alpha=0.70 / beta=0.30** — 10/10 share at least one genre; 9/10 distinct songs; mean S_num 0.864, mean S_genre 0.915

| # | track | artist | genres | combined | S_num | S_genre |
|---|---|---|---|---|---|---|
| 1 | Todo Bien | Lalo Ebratt;Juanes;Skinny Happy;Yera;Trapical | latin, latino, reggae +1 | 0.916 | 0.880 | 1.000 |
| 2 | Ram Pam Pam | Natti Natasha;Becky G | latin, latino, reggae +1 | 0.903 | 0.861 | 1.000 |
| 3 | Quizas - Remix | Tony Dize;Ken-Y;Marioso | latin, latino, reggae +1 | 0.890 | 0.843 | 1.000 |
| 4 | Quizas - Remix | Tony Dize;Ken-Y;Marioso | latin, latino, reggae +1 | 0.890 | 0.843 | 1.000 |
| 5 | La Copa | Ozuna | latin, latino, reggae +1 | 0.889 | 0.842 | 1.000 |
| 6 | Si Veo a Tu Mamá | Bad Bunny | latino, reggaeton | 0.868 | 0.937 | 0.707 |
| 7 | EL BARCO | KAROL G | reggae, reggaeton | 0.863 | 0.930 | 0.707 |
| 8 | Happy Happy | Nacho;Los Mendoza | latin, latino, reggae +1 | 0.862 | 0.803 | 1.000 |
| 9 | Desacatao (Remix) | Carlitos Junior;Yiordano Ignacio;BlackRoy | latino, reggae, reggaeton | 0.859 | 0.856 | 0.866 |
| 10 | Ponte Ahi | Casa De Leones | latino, reggae, reggaeton | 0.855 | 0.851 | 0.866 |

**alpha=0.90 / beta=0.10** — 5/10 share at least one genre; 10/10 distinct songs; mean S_num 0.958, mean S_genre 0.341

| # | track | artist | genres | combined | S_num | S_genre |
|---|---|---|---|---|---|---|
| 1 | Monotonía | Shakira;Ozuna | latin | 0.920 | 0.967 | 0.500 |
| 2 | Si Veo a Tu Mamá | Bad Bunny | latino, reggaeton | 0.914 | 0.937 | 0.707 |
| 3 | EL BARCO | KAROL G | reggae, reggaeton | 0.907 | 0.930 | 0.707 |
| 4 | Asi yo ladro | El Perro Chocolo | latino | 0.903 | 0.948 | 0.500 |
| 5 | Cê Tá Preparada - Ao Vivo | Tayrone | forro | 0.895 | 0.994 | 0.000 |
| 6 | Todo Bien | Lalo Ebratt;Juanes;Skinny Happy;Yera;Trapical | latin, latino, reggae +1 | 0.892 | 0.880 | 1.000 |
| 7 | Jalan Serta Yesus | Maranatha! Kids | children | 0.884 | 0.982 | 0.000 |
| 8 | La Boa | Alquimia La Sonora Del XXI | salsa | 0.882 | 0.980 | 0.000 |
| 9 | Eu Te Amo Demais | Bonde do gato preto | forro | 0.881 | 0.979 | 0.000 |
| 10 | Pon Mi | Shenseea;Dunw3ll | dancehall | 0.881 | 0.979 | 0.000 |

### most multi-genre (9 genres) — *Baby Blue - Remastered 2010* (blues, country, folk, j-pop +5)

**alpha=0.50 / beta=0.50** — 10/10 share at least one genre; 9/10 distinct songs; mean S_num 0.783, mean S_genre 0.588

| # | track | artist | genres | combined | S_num | S_genre |
|---|---|---|---|---|---|---|
| 1 | Takin' Care Of Business | Bachman-Turner Overdrive | blues, country, folk +2 | 0.724 | 0.702 | 0.745 |
| 2 | 靴ひも | Mr.Children | j-pop, j-rock | 0.691 | 0.910 | 0.471 |
| 3 | My Sweet Lord - Remastered 2014 | George Harrison | folk, singer-songwriter, songwriter | 0.689 | 0.800 | 0.577 |
| 4 | When You're Gone | Bryan Adams;Melanie C | singer-songwriter, songwriter | 0.688 | 0.906 | 0.471 |
| 5 | Abracadabra - Remastered 2017 | Steve Miller Band | blues, country, folk +4 | 0.681 | 0.606 | 0.756 |
| 6 | Abracadabra - Remastered 2017 | Steve Miller Band | blues, country, folk +4 | 0.681 | 0.606 | 0.756 |
| 7 | Voh | Prateek Kuhad | folk, singer-songwriter, songwriter | 0.678 | 0.779 | 0.577 |
| 8 | BAD FEELING | BOØWY | j-pop, j-rock, rockabilly | 0.675 | 0.966 | 0.385 |
| 9 | SCREAM | GLAY;EXILE | j-pop, j-rock | 0.674 | 0.877 | 0.471 |
| 10 | When You're In Love With A Beautiful Woman - Remastered 1996 | Dr. Hook | country, folk, singer-songwriter +1 | 0.671 | 0.676 | 0.667 |

**alpha=0.70 / beta=0.30** — 10/10 share at least one genre; 9/10 distinct songs; mean S_num 0.925, mean S_genre 0.390

| # | track | artist | genres | combined | S_num | S_genre |
|---|---|---|---|---|---|---|
| 1 | BAD FEELING | BOØWY | j-pop, j-rock, rockabilly | 0.792 | 0.966 | 0.385 |
| 2 | Merry-Go-Round | MAN WITH A MISSION | anime, j-pop, j-rock | 0.780 | 0.950 | 0.385 |
| 3 | 靴ひも | Mr.Children | j-pop, j-rock | 0.778 | 0.910 | 0.471 |
| 4 | When You're Gone | Bryan Adams;Melanie C | singer-songwriter, songwriter | 0.775 | 0.906 | 0.471 |
| 5 | Confetti - Remastered | The Lemonheads | power-pop | 0.767 | 0.953 | 0.333 |
| 6 | 世界は恋に落ちている | CHiCO with HoneyWorks | anime, j-pop, j-rock | 0.759 | 0.920 | 0.385 |
| 7 | SCREAM | GLAY;EXILE | j-pop, j-rock | 0.755 | 0.877 | 0.471 |
| 8 | California Dreamin' - 2007 Digital Remaster | The Beach Boys | psych-rock | 0.753 | 0.932 | 0.333 |
| 9 | Clarity | Vance Joy | folk | 0.744 | 0.920 | 0.333 |
| 10 | Clarity | Vance Joy | folk | 0.744 | 0.920 | 0.333 |

**alpha=0.90 / beta=0.10** — 4/10 share at least one genre; 10/10 distinct songs; mean S_num 0.960, mean S_genre 0.144

| # | track | artist | genres | combined | S_num | S_genre |
|---|---|---|---|---|---|---|
| 1 | BAD FEELING | BOØWY | j-pop, j-rock, rockabilly | 0.908 | 0.966 | 0.385 |
| 2 | Merry-Go-Round | MAN WITH A MISSION | anime, j-pop, j-rock | 0.893 | 0.950 | 0.385 |
| 3 | Confetti - Remastered | The Lemonheads | power-pop | 0.891 | 0.953 | 0.333 |
| 4 | It's My Life | Dr. Alban | house | 0.879 | 0.976 | 0.000 |
| 5 | Take a Chance On Me | Erasure | synth-pop | 0.873 | 0.970 | 0.000 |
| 6 | California Dreamin' - 2007 Digital Remaster | The Beach Boys | psych-rock | 0.872 | 0.932 | 0.333 |
| 7 | Dejar de ser | A.N.I.M.A.L.;Chuck Jhonson and Juan Jose Burgos;Richard Kaplan | heavy-metal | 0.869 | 0.966 | 0.000 |
| 8 | Untraveled Road | Thousand Foot Krutch | metal | 0.868 | 0.964 | 0.000 |
| 9 | Adidas & Pearls | Solardo;Vintage Culture;LOWES | deep-house, groove | 0.867 | 0.964 | 0.000 |
| 10 | Bolingo - Original Short Edit | Ferreck Dawn | deep-house, disco, groove | 0.866 | 0.963 | 0.000 |

## How much the list changes with the weight

Jaccard overlap of the top-10 sets (|intersection| / |union|) against alpha=0.50,
and against the previous alpha.

| reference | 0.50 vs 0.60 | 0.50 vs 0.70 | 0.50 vs 0.80 | 0.50 vs 0.90 | 0.50 set vs 0.90 set (shared) |
|---|---|---|---|---|---|
| single-genre rock (popular) | 0.67 | 0.54 | 0.33 | 0.18 | 3/10 |
| single-genre hip-hop (popular) | 1.00 | 0.82 | 0.82 | 0.05 | 1/10 |
| single-genre classical | 1.00 | 1.00 | 1.00 | 1.00 | 10/10 |
| single-genre techno | 1.00 | 0.82 | 0.67 | 0.25 | 4/10 |
| multi-genre (>=4 genres) | 1.00 | 0.43 | 0.33 | 0.05 | 1/10 |
| most multi-genre (9 genres) | 0.43 | 0.25 | 0.18 | 0.05 | 1/10 |

## Does one block dominate? (sample of 300 tracks)

For each sampled reference the top-10 was computed at beta-heavy (0.5/0.5) and
numeric-heavy (0.9/0.1) settings, then the *composition of the returned list* was
measured. `genre-match` = share of the top-10 sharing at least one genre with the
reference.

| setting | mean genre-match in top-10 | mean S_num of top-10 | mean S_genre of top-10 |
|---|---|---|---|
| alpha=0.50 | 1.000 | 0.899 | 0.980 |
| alpha=0.70 | 1.000 | 0.912 | 0.961 |
| alpha=0.90 | 0.839 | 0.946 | 0.776 |

## Observations

- The top-10 changes most for **single-genre hip-hop (popular)** (1/10 tracks survive from
  alpha=0.50 to 0.90) and least for **single-genre classical** (10/10).
- Across the 300 sampled references, genre-match in the top-10 rises from
  **1.000** at 0.5/0.5 to **0.839** at 0.9/0.1.
- Mean S_num of the returned list moves from **0.899** (0.5/0.5) to **0.946** (0.9/0.1), while mean S_genre
  moves from **0.980** to **0.776**.

### Does one block dominate?

- **At 0.5/0.5 and 0.7/0.3 the genre block effectively decides the list**: 100% and 100% of sampled top-10 results share a genre with the reference. A genre weight of only 0.3 already makes nearly every neighbour same-genre.
- **Only at 0.9/0.1 does audio similarity visibly lead**: genre-match falls to 84% and cross-genre results appear (e.g. a house track beside a multi-genre reference with S_genre = 0.000).
- This is structural rather than a bug: two tracks in the same genre have S_genre = 1.0 exactly, while two very close audio profiles rarely exceed S_num ≈ 0.95, so genre wins whenever its weight is comparable.
- One reference (*classical*) returns an identical top-10 at every weight, because its nearest tracks are close in **both** blocks; re-weighting cannot change a list whose candidates already lead in both.

### Defect found: duplicate songs inside one list

- A rock top-10 contains *Follow You* twice: 4,647 `(track_name, artists)` groups hold more than one `track_id`. De-duplication by `track_id` therefore does **not** yet satisfy the brief's no-duplicate-results rule.
- Phase 5 needs an explicit duplicate-suppression rule, and it must not be silent: 2,316 same-title groups differ in audio features and may be genuinely different versions (live/remaster).

- No weight is 'correct': without relevance labels there is no ground truth to optimise
  against, and the highest-scoring list is not necessarily the one a listener prefers.
  The choice is a documented design decision, to be defended qualitatively and with the
  structural checks planned in Phase 6.

## Exclusions applied

- **Self-match:** each reference row is removed before ranking (it would score 1.0).
- **Duplicates:** the 24,259 repeated-`track_id` rows are gone (Phase 4), and the 1 junk row plus 157 zero-coded tracks are absent from the catalogue. **However**, the same recording under two different `track_id`s is *not* removed — see the duplicate section above; that is a known defect to resolve in Phase 5, reported rather than silently patched here.

