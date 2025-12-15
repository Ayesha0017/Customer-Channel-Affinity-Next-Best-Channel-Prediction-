
campaigns_meta = pd.read_csv(r"C:\Users\ayesh\OneDrive\Documents\Channel affinity\data\comprehensive_data\campaigns_meta.csv")
content_engagement = pd.read_csv(r"C:\Users\ayesh\OneDrive\Documents\Channel affinity\data\comprehensive_data\content_engagement.csv")
customers = pd.read_csv(r"C:\Users\ayesh\OneDrive\Documents\Channel affinity\data\comprehensive_data\customers.csv")
email_engagement = pd.read_csv(r"C:\Users\ayesh\OneDrive\Documents\Channel affinity\data\comprehensive_data\email_engagement.csv")
marketing_touchpoints = pd.read_csv(r"C:\Users\ayesh\OneDrive\Documents\Channel affinity\data\comprehensive_data\marketing_touchpoints.csv")
order_items = pd.read_csv(r"C:\Users\ayesh\OneDrive\Documents\Channel affinity\data\comprehensive_data\order_items.csv")
orders = pd.read_csv(r"C:\Users\ayesh\OneDrive\Documents\Channel affinity\data\comprehensive_data\orders.csv")
sessions = pd.read_csv(r"C:\Users\ayesh\OneDrive\Documents\Channel affinity\data\comprehensive_data\sessions.csv")
social_media_engagement = pd.read_csv(r"C:\Users\ayesh\OneDrive\Documents\Channel affinity\data\comprehensive_data\social_media_engagement.csv")


# Rules-based Channel Affinity - Aggregation + Scoring (pandas)
# Assumes you have the tables loaded as pandas DataFrames with the names:
# campaigns_meta, content_engagement, customers, email_engagement,
# marketing_touchpoints, orders, order_items, sessions, social_media_engagement

import pandas as pd
import numpy as np

# ---------- CONFIG ----------
# Channels list (as used across your schema)
CHANNELS = [
    "Organic Search", "YouTube", "Direct", "Facebook", "Email",
    "Instagram", "Google Ads"
]

# Define analysis 'as of' date (end of dataset window)
AS_OF = pd.to_datetime("2024-01-01")

# Weights by channel for features (these are business-rule weights - adjustable)
# Each channel score = sum(feature_normalized * weight)
CHANNEL_WEIGHTS = {
    "Email": {
        "email_open_rate": 0.45,
        "email_click_rate": 0.25,
        "time_to_open_inv": 0.15,    # faster open => better
        "recent_order_recency_inv": 0.10,
        "revenue_attributed_email": 0.05
    },
    "YouTube": {
        "video_watch_seconds": 0.6,
        "video_completion_pct": 0.2,
        "social_youtube_engagements": 0.1,
        "revenue_attributed_youtube": 0.1
    },
    "Google Ads": {
        "touchpoint_impressions_ga": 0.25,
        "touchpoint_clicks_ga": 0.25,
        "sessions_engagement_score_ga": 0.2,
        "revenue_attributed_ga": 0.2,
        "acquired_via_ga": 0.1
    },
    "Facebook": {
        "social_fb_engagements": 0.35,
        "social_sentiment_pos_fb": 0.15,
        "touchpoint_clicks_fb": 0.2,
        "revenue_attributed_fb": 0.2,
        "sessions_engagement_score_fb": 0.1
    },
    "Instagram": {
        "social_ig_engagements": 0.4,
        "touchpoint_clicks_ig": 0.25,
        "revenue_attributed_ig": 0.2,
        "sessions_engagement_score_ig": 0.15
    },
    "Direct": {
        "sessions_direct_engagement": 0.5,
        "recent_order_recency_inv": 0.2,
        "revenue_direct": 0.2,
        "acquired_via_direct": 0.1
    },
    "Organic Search": {
        "sessions_org_engagement": 0.45,
        "touchpoint_impressions_org": 0.25,
        "revenue_org": 0.2,
        "acquired_via_organic": 0.1
    }
}

# ---------- HELPERS ----------
def safe_minmax_series(s):
    """Min-max normalize a pandas Series safely (if constant, return zeros)."""
    if s.isna().all():
        return s.fillna(0)
    mn = s.min()
    mx = s.max()
    if mn == mx:
        return pd.Series(0, index=s.index)
    return (s - mn) / (mx - mn)

# ---------- PREPROCESS NULLS ----------
# Impute last_purchase_date nulls -> keep NaT but create recency feature with large value
customers = customers.copy()
customers['acquisition_date'] = pd.to_datetime(customers['acquisition_date'])
customers['last_purchase_date'] = pd.to_datetime(customers['last_purchase_date'])

# For time_to_open_minutes nulls will appear in email_engagement; we'll impute with median later
email_engagement = email_engagement.copy()
email_engagement['event_timestamp'] = pd.to_datetime(email_engagement['event_timestamp'])
email_engagement['time_to_open_minutes'] = pd.to_numeric(email_engagement['time_to_open_minutes'], errors='coerce')

# ---------- AGGREGATE: customers baseline ----------
cust_base = customers[['customer_id', 'acquisition_channel', 'acquisition_date', 'total_revenue', 'total_orders', 'avg_order_value', 'customer_segment', 'customer_status', 'lifetime_value']].copy()
# recency (days since last purchase) - if null -> large recency
cust_base['last_purchase_date'] = customers['last_purchase_date']
cust_base['recency_days'] = (AS_OF - cust_base['last_purchase_date']).dt.days
cust_base['recency_days'] = cust_base['recency_days'].fillna((AS_OF - cust_base['acquisition_date']).dt.days + 999)  # if never purchased, make recency large

# inverse recency for scoring (more recent -> higher)
cust_base['recency_inv'] = 1 / (1 + cust_base['recency_days'])

# ---------- AGGREGATE: email_engagement ----------
# Focus on events: opened, click counts, time to open distribution -> per customer
email = email_engagement.copy()
# Consider only relevant event_types (opened, delivered, sent) — we compute open rate & click rate
# We'll compute: total_sent, total_opened (event_type == 'opened'), total_clicks (click_count), avg_time_to_open (minutes)
email_stats = email.groupby('customer_id').agg(
    total_sent = ('event_type', lambda x: (x == 'sent').sum()),
    total_opened = ('event_type', lambda x: (x == 'opened').sum()),
    total_clicks = ('click_count', 'sum'),
    avg_time_to_open = ('time_to_open_minutes', lambda s: s.dropna().median() if s.dropna().shape[0]>0 else np.nan)
).reset_index()

email_stats['email_open_rate'] = email_stats['total_opened'] / email_stats['total_sent'].replace(0, np.nan)
email_stats['email_click_rate'] = email_stats['total_clicks'] / email_stats['total_sent'].replace(0, np.nan)

# Impute avg_time_to_open median for missing
median_tto = email_stats['avg_time_to_open'].median()
email_stats['avg_time_to_open'] = email_stats['avg_time_to_open'].fillna(median_tto)
# faster open is better -> create inverse (bounded)
email_stats['time_to_open_inv'] = 1 / (1 + email_stats['avg_time_to_open'])

# replace NaN open/click rates with 0
email_stats['email_open_rate'] = email_stats['email_open_rate'].fillna(0)
email_stats['email_click_rate'] = email_stats['email_click_rate'].fillna(0)

# ---------- AGGREGATE: content_engagement (video focus for YouTube) ----------
content = content_engagement.copy()
content['engagement_timestamp'] = pd.to_datetime(content['engagement_timestamp'])
# Identify video/watch-type engagements: use content_type or engagement_type 'watch' etc.
video_mask = (content['content_type'].str.lower().fillna('').str.contains('video')) | (content['engagement_type'].str.lower()=='watch')
content_video = content[video_mask].copy()
video_stats = content_video.groupby('customer_id').agg(
    video_watch_seconds = ('time_spent_seconds', 'sum'),
    video_engagements = ('engagement_id', 'count'),
    video_completion_avg = ('completion_percentage', 'mean')
).reset_index().fillna(0)

# ---------- AGGREGATE: social_media_engagement ----------
social = social_media_engagement.copy()
social['engagement_timestamp'] = pd.to_datetime(social['engagement_timestamp'])
# Count engagements by platform and positive sentiment proportion
social['is_positive'] = social['sentiment_score'] > 0.2  # threshold for positive
social_stats = social.groupby(['customer_id','platform']).agg(
    social_engagements = ('social_engagement_id','count'),
    social_positive = ('is_positive','sum')
).reset_index()

# Pivot to per-platform columns (FB, IG, YouTube may also appear)
social_pivot = social_stats.pivot(index='customer_id', columns='platform', values='social_engagements').fillna(0)
social_pos = social_stats.pivot(index='customer_id', columns='platform', values='social_positive').fillna(0)
# rename columns to predictable names if present
def safe_col(df, platform):
    return platform if platform in df.columns else None

# ---------- AGGREGATE: marketing_touchpoints ----------
mt = marketing_touchpoints.copy()
mt['timestamp'] = pd.to_datetime(mt['timestamp'])
# We'll aggregate by channel and also get revenue_attributed and cost
mt_by_cust_chan = mt.groupby(['customer_id','channel']).agg(
    tp_count = ('touchpoint_id','count'),
    tp_cost = ('cost','sum'),
    tp_revenue = ('revenue_attributed','sum'),
    tp_impressions = ('touchpoint_type', lambda s: (s=='impression').sum()),
    tp_clicks = ('touchpoint_type', lambda s: (s=='click').sum())
).reset_index()

# Pivot so we have per-customer per-channel columns for some features
mt_pivot_cost = mt_by_cust_chan.pivot(index='customer_id', columns='channel', values='tp_cost').fillna(0)
mt_pivot_rev = mt_by_cust_chan.pivot(index='customer_id', columns='channel', values='tp_revenue').fillna(0)
mt_pivot_clicks = mt_by_cust_chan.pivot(index='customer_id', columns='channel', values='tp_clicks').fillna(0)
mt_pivot_impr = mt_by_cust_chan.pivot(index='customer_id', columns='channel', values='tp_impressions').fillna(0)

# rename columns to consistent keys, replacing spaces with underscores for convenience
mt_pivot_cost.columns = [f"cost_{c.replace(' ','_')}" for c in mt_pivot_cost.columns]
mt_pivot_rev.columns = [f"rev_{c.replace(' ','_')}" for c in mt_pivot_rev.columns]
mt_pivot_clicks.columns = [f"clicks_{c.replace(' ','_')}" for c in mt_pivot_clicks.columns]
mt_pivot_impr.columns = [f"impr_{c.replace(' ','_')}" for c in mt_pivot_impr.columns]

# ---------- AGGREGATE: sessions (engagement_score by channel) ----------
sess = sessions.copy()
sess['session_start_timestamp'] = pd.to_datetime(sess['session_start_timestamp'])
sess['session_end_timestamp'] = pd.to_datetime(sess['session_end_timestamp'])
sess_agg_by_cust_channel = sess.groupby(['customer_id','channel']).agg(
    sessions_count = ('session_id','count'),
    avg_engagement_score = ('engagement_score','mean'),
    avg_session_duration = ('session_duration_seconds','mean'),
    avg_bounce_rate = ('bounce_rate','mean')
).reset_index()

# pivot sessions engagement score per channel
sess_pivot_eng = sess_agg_by_cust_channel.pivot(index='customer_id', columns='channel', values='avg_engagement_score').fillna(0)
sess_pivot_sessions = sess_agg_by_cust_channel.pivot(index='customer_id', columns='channel', values='sessions_count').fillna(0)
# rename
sess_pivot_eng.columns = [f"engagement_score_{c.replace(' ','_')}" for c in sess_pivot_eng.columns]
sess_pivot_sessions.columns = [f"sessions_count_{c.replace(' ','_')}" for c in sess_pivot_sessions.columns]

# ---------- AGGREGATE: orders (revenue by customer, optionally by channel if attribution exists) ----------
orders_ = orders.copy()
orders_['order_timestamp'] = pd.to_datetime(orders_['order_timestamp'])
order_revenue = orders_.groupby('customer_id').agg(
    total_order_value = ('order_value','sum'),
    orders_count = ('order_id','count'),
    last_order_date = ('order_timestamp','max')
).reset_index()

# merge order items for product-level analysis if needed (skipped for now)

# ---------- MERGE ALL AGGREGATES to single customer dataframe ----------
# ---------- MERGE ALL AGGREGATES to single customer dataframe (fixed) ----------
# Prepare list of (df, prefix) so we can add prefixes to avoid overlapping column names.
# Keep cust_base unprefixed so its core fields are readable.
merge_list = [
    (cust_base.set_index('customer_id'), None),          # keep as is
    (email_stats.set_index('customer_id'), 'email'),
    (video_stats.set_index('customer_id'), 'video'),
    (order_revenue.set_index('customer_id'), 'orders'),
    (mt_pivot_cost, 'mt_cost'),
    (mt_pivot_rev, 'mt_rev'),
    (mt_pivot_clicks, 'mt_clicks'),
    (mt_pivot_impr, 'mt_impr'),
    (sess_pivot_eng, 'sess_eng'),
    (sess_pivot_sessions, 'sess_cnt'),
    (social_pivot, 'social'),
    (social_pos, 'socialpos')
]

# Start from cust_base (preserves its readable column names)
cust_agg = merge_list[0][0].copy()

# Join each other df after prefixing their columns (if a prefix is given)
for df, prefix in merge_list[1:]:
    if prefix:
        # ensure df index is customer_id
        df = df.copy()
        df.index.name = 'customer_id'
        # add prefix to all columns to avoid overlap
        df = df.add_prefix(prefix + "_")
    # left-join on index (customer_id)
    cust_agg = cust_agg.join(df, how='left')

# fill NaNs with zeros for numeric columns
numeric_cols = cust_agg.select_dtypes(include=[np.number]).columns
cust_agg[numeric_cols] = cust_agg[numeric_cols].fillna(0)

# If you need any of the prefixed columns accessible in norm_features later,
# update the feature names mapping (or the code that looks up columns). 
# For example, rev_Google_Ads earlier came from mt_pivot_rev and is now named 'mt_rev_rev_Google_Ads'


# ---------- NORMALIZE candidate features used in scoring (FIXED) ----------
# Utility to find the first existing column from a list of candidates in cust_agg
def find_col(*cands):
    for c in cands:
        if c in cust_agg.columns:
            return c
    return None

norm_features = {}

# Email features (prefixed names become 'email_*' after join)
email_open_col = find_col('email_open_rate', 'email_email_open_rate')
email_click_col = find_col('email_click_rate', 'email_email_click_rate')
time_to_open_col = find_col('time_to_open_inv', 'email_time_to_open_inv', 'email_avg_time_to_open')

norm_features['email_open_rate'] = safe_minmax_series(cust_agg[email_open_col]) if email_open_col else pd.Series(0, index=cust_agg.index)
norm_features['email_click_rate'] = safe_minmax_series(cust_agg[email_click_col]) if email_click_col else pd.Series(0, index=cust_agg.index)
norm_features['time_to_open_inv'] = safe_minmax_series(cust_agg[time_to_open_col]) if time_to_open_col else pd.Series(0, index=cust_agg.index)

# Recency inverse (cust_base kept unprefixed)
norm_features['recent_order_recency_inv'] = safe_minmax_series(cust_agg.get('recency_inv', pd.Series(0, index=cust_agg.index)))

# revenue attributed per channel: search for prefixed mt_rev_rev_<Channel> OR rev_<Channel>
for ch in CHANNELS:
    ckey = ch.replace(' ','_')
    candidates = [
        f"mt_rev_rev_{ch}",
        f"mt_rev_rev_{ckey}",
        f"rev_{ch.replace(' ','_')}",
        f"mt_rev_rev_{ch.replace(' ','_')}",
        f"mt_rev_rev_{ckey}"
    ]
    col = find_col(*candidates)
    norm_features[f"revenue_attributed_{ckey}"] = safe_minmax_series(cust_agg[col]) if col else pd.Series(0, index=cust_agg.index)

# Video / YouTube (these got prefixed as 'video_*')
video_watch_col = find_col('video_video_watch_seconds','video_watch_seconds','video_video_watch_seconds')
video_completion_col = find_col('video_video_completion_avg','video_completion_avg','video_video_completion_avg')
video_eng_col = find_col('video_video_engagements','video_engagements')

norm_features['video_watch_seconds'] = safe_minmax_series(cust_agg[video_watch_col]) if video_watch_col else pd.Series(0, index=cust_agg.index)
norm_features['video_completion_pct'] = safe_minmax_series(cust_agg[video_completion_col]) if video_completion_col else pd.Series(0, index=cust_agg.index)
norm_features['video_engagements'] = safe_minmax_series(cust_agg[video_eng_col]) if video_eng_col else pd.Series(0, index=cust_agg.index)

# Sessions & touchpoint features per channel
for ch in CHANNELS:
    ckey = ch.replace(' ','_')
    # engagement score candidates: sess_eng_engagement_score_<Channel> OR engagement_score_<Channel>
    eng_candidates = [
        f"sess_eng_engagement_score_{ch}", f"sess_eng_engagement_score_{ckey}",
        f"engagement_score_{ch.replace(' ','_')}", f"engagement_score_{ckey}"
    ]
    eng_col = find_col(*eng_candidates)
    norm_features[f"sessions_engagement_score_{ckey}"] = safe_minmax_series(cust_agg[eng_col]) if eng_col else pd.Series(0, index=cust_agg.index)

    # clicks/impr candidates (mt_clicks_clicks_<Channel>, mt_impr_impr_<Channel>, clicks_<Channel>, impr_<Channel>)
    clicks_candidates = [
        f"mt_clicks_clicks_{ch}", f"mt_clicks_clicks_{ckey}",
        f"clicks_{ch.replace(' ','_')}", f"clicks_{ckey}"
    ]
    impr_candidates = [
        f"mt_impr_impr_{ch}", f"mt_impr_impr_{ckey}",
        f"impr_{ch.replace(' ','_')}", f"impr_{ckey}"
    ]
    clicks_col = find_col(*clicks_candidates)
    impr_col = find_col(*impr_candidates)
    norm_features[f"tp_clicks_{ckey}"] = safe_minmax_series(cust_agg[clicks_col]) if clicks_col else pd.Series(0, index=cust_agg.index)
    norm_features[f"tp_impr_{ckey}"] = safe_minmax_series(cust_agg[impr_col]) if impr_col else pd.Series(0, index=cust_agg.index)

    # revenue normalized alias (already computed above as revenue_attributed_<ckey>)
    norm_features[f"rev_{ckey}"] = norm_features.get(f"revenue_attributed_{ckey}", pd.Series(0, index=cust_agg.index))

# Social platform engagements: columns will be 'social_<Platform>' or 'socialpos_<Platform>'
for platform in ['Facebook','Instagram','YouTube','Google Ads','Direct','Organic Search']:
    platform_key = platform.replace(' ','_')
    col_social = find_col(f"social_{platform}", f"social_{platform_key}", f"social_{platform.replace(' ','_')}", f"social_{platform_key}")
    col_socialpos = find_col(f"socialpos_{platform}", f"socialpos_{platform_key}", f"socialpos_{platform.replace(' ','_')}", f"socialpos_{platform_key}")
    norm_features[f"social_{platform_key}_eng"] = safe_minmax_series(cust_agg[col_social]) if col_social else pd.Series(0, index=cust_agg.index)
    # positive counts if needed
    norm_features[f"social_{platform_key}_pos"] = safe_minmax_series(cust_agg[col_socialpos]) if col_socialpos else pd.Series(0, index=cust_agg.index)

# ---------- COMPUTE CHANNEL SCORES (unchanged logic, but will now find correct norm_features) ----------
scores = pd.DataFrame(index=cust_agg.index)

for ch in CHANNELS:
    ch_key = ch.replace(' ','_')
    weights = CHANNEL_WEIGHTS.get(ch, {})
    score = pd.Series(0.0, index=cust_agg.index)
    for feat_name, w in weights.items():
        if feat_name.startswith('revenue_attributed'):
            norm_key = f"revenue_attributed_{ch_key}"
            val = norm_features.get(norm_key, pd.Series(0, index=cust_agg.index))
        else:
            # direct match in norm_features
            if feat_name in norm_features:
                val = norm_features[feat_name]
            else:
                # try channel-suffixed pattern
                trial = f"{feat_name}_{ch_key}"
                val = norm_features.get(trial, pd.Series(0, index=cust_agg.index))
        score = score + (val * w)
    scores[f"score_{ch_key}"] = safe_minmax_series(score)

# preferred channel = argmax of scores
scores['preferred_channel'] = scores.idxmax(axis=1).str.replace('score_','').str.replace('_',' ')
scores['preferred_channel'] = scores['preferred_channel'].replace({
    'Google Ads': 'Google Ads',
    'Organic Search': 'Organic Search'
})


# create final output table
channel_affinity = scores.reset_index()[['customer_id'] + [c for c in scores.columns]]

# OPTIONAL: attach some identifying customer info
channel_affinity = channel_affinity.merge(customers[['customer_id','acquisition_channel','customer_segment']], on='customer_id', how='left')

# Save or preview top rows
print(channel_affinity.head())

# channel_affinity now contains per-channel normalized scores and 'preferred_channel' for each customer.
# Tweak CHANNEL_WEIGHTS to align with your business priorities. Also you can restrict features to last N days by
# filtering timestamps before aggregation.
# Save final channel affinity output
channel_affinity.to_csv(r"C:\Users\ayesh\OneDrive\Documents\Channel affinity\output\channel_affinity_results.csv", index=False)
print("File saved as channel_affinity_results.csv")

