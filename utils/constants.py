INPUT_CSV_FILEPATH = "data/threads_jul24_mar25.csv"
OUTPUT_CSV_FILEPATH = "data/model_comparison_results.csv"
OUTPUT_EXCEL_FILEPATH = "data/model_comparison_results.xlsx"
VALIDATION_FILEPATH = "data/threads_jul24_mar25_val.csv"
TRAIN_FILEPATH = "data/threads_jul24_mar25_train.csv"
TEST_FILEPATH = "data/threads_jul24_mar25_test.csv"
TIME_COLUMNS = [
    "time_of_day_low_upvotes",
    "time_of_day_high_upvotes",
]

SENTIMENT_COLUMNS = [
    "sentiment_positive",
    "sentiment_negative",
]

BOOLEAN_COLUMNS = (
    [
        "is_requesting_help",
        "is_relationship",
        "over_18",
    ]
    + TIME_COLUMNS
    + SENTIMENT_COLUMNS
)
CYCLICAL_COLUMNS = ["hour_sin", "hour_cos"]
ORDINAL_COLUMNS = ["word_count_bins"]

SCALER_FILENAME = "scaler.pkl"
# NOTE: does not contain "f_43_type_token", "f_44_mean_word_length" as some rows have missing data
BIBER_FEATURES = [
    "f_01_past_tense",
    "f_02_perfect_aspect",
    "f_03_present_tense",
    "f_04_place_adverbials",
    "f_05_time_adverbials",
    "f_06_first_person_pronouns",
    "f_07_second_person_pronouns",
    "f_08_third_person_pronouns",
    "f_09_pronoun_it",
    "f_10_demonstrative_pronoun",
    "f_11_indefinite_pronouns",
    "f_12_proverb_do",
    "f_13_wh_question",
    "f_14_nominalizations",
    "f_15_gerunds",
    "f_16_other_nouns",
    "f_17_agentless_passives",
    "f_18_by_passives",
    "f_19_be_main_verb",
    "f_20_existential_there",
    "f_21_that_verb_comp",
    "f_22_that_adj_comp",
    "f_23_wh_clause",
    "f_24_infinitives",
    "f_25_present_participle",
    "f_26_past_participle",
    "f_27_past_participle_whiz",
    "f_28_present_participle_whiz",
    "f_29_that_subj",
    "f_30_that_obj",
    "f_31_wh_subj",
    "f_32_wh_obj",
    "f_33_pied_piping",
    "f_34_sentence_relatives",
    "f_35_because",
    "f_36_though",
    "f_37_if",
    "f_38_other_adv_sub",
    "f_39_prepositions",
    "f_40_adj_attr",
    "f_41_adj_pred",
    "f_42_adverbs",
    "f_45_conjuncts",
    "f_46_downtoners",
    "f_47_hedges",
    "f_48_amplifiers",
    "f_49_emphatics",
    "f_50_discourse_particles",
    "f_51_demonstratives",
    "f_52_modal_possibility",
    "f_53_modal_necessity",
    "f_54_modal_predictive",
    "f_55_verb_public",
    "f_56_verb_private",
    "f_57_verb_suasive",
    "f_58_verb_seem",
    "f_59_contractions",
    "f_60_that_deletion",
    "f_61_stranded_preposition",
    "f_62_split_infinitive",
    "f_63_split_auxiliary",
    "f_64_phrasal_coordination",
    "f_65_clausal_coordination",
    "f_66_neg_synthetic",
    "f_67_neg_analytic",
]

ALL_FEATURES = (
    [
        "V_Mean",
        "A_Mean",
        "D_Mean",
        "title_rating",
        "content_rating",
    ]
    + BOOLEAN_COLUMNS
    + BIBER_FEATURES
    + ORDINAL_COLUMNS
)


POSITIVE_PERMUTATION_FEATURES = [
    "is_requesting_help",
    "title_rating",
    "content_rating",
    "is_relationship",
    "f_19_be_main_verb",
    "f_52_modal_possibility",
    "f_51_demonstratives",
    "f_39_prepositions",
    "f_58_verb_seem",
    "f_03_present_tense",
    "f_49_emphatics",
    "f_12_proverb_do",
    "f_42_adverbs",
    "f_15_gerunds",
    "f_20_existential_there",
    "f_59_contractions",
    "f_08_third_person_pronouns",
    "f_01_past_tense",
    "f_02_perfect_aspect",
    "f_17_agentless_passives",
    "f_38_other_adv_sub",
    "f_07_second_person_pronouns",
    "D_Mean",
    "f_35_because",
    "f_48_amplifiers",
    "over_18",
    "f_23_wh_clause",
    "f_61_stranded_preposition",
    "f_57_verb_suasive",
    "A_Mean",
    "f_47_hedges",
    "sentiment_negative",
    "f_05_time_adverbials",
    "f_54_modal_predictive",
    "f_32_wh_obj",
    "time_of_day_low_upvotes",
    "f_56_verb_private",
    "f_37_if",
    "f_24_infinitives",
    "f_06_first_person_pronouns",
    "f_36_though",
    "f_29_that_subj",
    "word_count_bins",
    "f_63_split_auxiliary",
    "f_10_demonstrative_pronoun",
    "f_50_discourse_particles",
    "f_28_present_participle_whiz",
    "f_30_that_obj",
    "f_65_clausal_coordination",
    "f_46_downtoners",
]
