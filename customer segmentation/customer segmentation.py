"""
CUSTOMER SEGMENTATION SYSTEM
================================
This system segments customers based on purchasing behavior using RFM analysis
(Recency, Frequency, Monetary value) and K-Means clustering.
Output File: customer_segmentation.csv
"""

# ============================================
# STEP 1: IMPORT LIBRARIES
# ============================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import warnings
import os
warnings.filterwarnings('ignore')

# Set visualization style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
print("="*60)
print("CUSTOMER SEGMENTATION SYSTEM")
print("="*60)
print("✅ Libraries imported successfully")

# ============================================
# STEP 2: DATA LOADING
# ============================================
print("\n" + "="*60)
print("STEP 2: LOADING DATASET")
print("="*60)

# Use your CSV file path
file_path = r"C:\Users\Dell\customer segmentation\online_retail_II.csv"
print(f"📂 Loading dataset from: {file_path}")

try:
    # For CSV files, we need to specify encoding and handle dates
    df = pd.read_csv(file_path, encoding='ISO-8859-1')
    print(f"✅ Dataset loaded successfully!")
    print(f"   Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"   Columns: {list(df.columns)}")
except FileNotFoundError:
    print(f"❌ File not found at: {file_path}")
    print("Please check if the file exists in that location")
    exit()
except Exception as e:
    print(f"❌ Error loading file: {e}")
    exit()

# ============================================
# STEP 3: INITIAL DATA EXPLORATION
# ============================================
print("\n" + "="*60)
print("STEP 3: DATA EXPLORATION")
print("="*60)

print("\n📊 First 5 rows:")
print(df.head())

print("\n📊 Dataset Info:")
print(df.info())

print("\n📊 Basic Statistics:")
print(df.describe())

# Check column names (they might be different in CSV)
print("\n📊 Column Names:")
for i, col in enumerate(df.columns):
    print(f"   {i+1}. {col}")

# ============================================
# STEP 4: DATA CLEANING
# ============================================
print("\n" + "="*60)
print("STEP 4: DATA CLEANING")
print("="*60)

# Make a copy to avoid modifying original
cleaned_df = df.copy()
initial_rows = len(cleaned_df)

# Standardize column names (they might be slightly different in CSV)
column_mapping = {}
for col in cleaned_df.columns:
    col_lower = col.lower().strip()
    if 'invoice' in col_lower and 'date' not in col_lower:
        column_mapping[col] = 'Invoice'
    elif 'stock' in col_lower:
        column_mapping[col] = 'StockCode'
    elif 'description' in col_lower:
        column_mapping[col] = 'Description'
    elif 'quantity' in col_lower:
        column_mapping[col] = 'Quantity'
    elif 'invoice' in col_lower and 'date' in col_lower:
        column_mapping[col] = 'InvoiceDate'
    elif 'price' in col_lower or 'unit' in col_lower:
        column_mapping[col] = 'Price'
    elif 'customer' in col_lower or 'id' in col_lower:
        column_mapping[col] = 'Customer ID'
    elif 'country' in col_lower:
        column_mapping[col] = 'Country'

# Rename columns if mapping exists
if column_mapping:
    cleaned_df.rename(columns=column_mapping, inplace=True)
    print("✅ Standardized column names")
    print(f"   New columns: {list(cleaned_df.columns)}")

# Convert InvoiceDate to datetime
if 'InvoiceDate' in cleaned_df.columns:
    try:
        cleaned_df['InvoiceDate'] = pd.to_datetime(cleaned_df['InvoiceDate'])
        print("✅ Converted InvoiceDate to datetime")
    except:
        print("⚠️ Could not convert InvoiceDate to datetime")

# Convert to string for text columns
for col in ['Invoice', 'StockCode', 'Customer ID']:
    if col in cleaned_df.columns:
        cleaned_df[col] = cleaned_df[col].astype(str)
        print(f"✅ Converted {col} to string")

# Remove rows with missing Customer IDs
if 'Customer ID' in cleaned_df.columns:
    initial_with_na = len(cleaned_df)
    cleaned_df = cleaned_df[cleaned_df['Customer ID'] != 'nan']
    cleaned_df = cleaned_df[cleaned_df['Customer ID'] != '']
    print(f"✅ Removed rows with missing Customer IDs: {initial_with_na - len(cleaned_df)} rows removed")

# Remove invalid quantities and prices
if 'Quantity' in cleaned_df.columns and 'Price' in cleaned_df.columns:
    cleaned_df = cleaned_df[cleaned_df['Quantity'] > 0]
    cleaned_df = cleaned_df[cleaned_df['Price'] > 0]
    print(f"✅ Removed invalid quantities/prices: {initial_rows - len(cleaned_df)} total rows removed")

print(f"\n📊 Cleaned dataset shape: {cleaned_df.shape}")

# ============================================
# STEP 5: FEATURE ENGINEERING (RFM)
# ============================================
print("\n" + "="*60)
print("STEP 5: RFM FEATURE ENGINEERING")
print("="*60)

# Create total sales column
if 'Quantity' in cleaned_df.columns and 'Price' in cleaned_df.columns:
    cleaned_df['TotalSales'] = cleaned_df['Quantity'] * cleaned_df['Price']
    print("✅ Created TotalSales column")

# Group by Customer ID
if 'Customer ID' in cleaned_df.columns:
    rfm_df = cleaned_df.groupby('Customer ID').agg({
        'TotalSales': 'sum',  # Monetary
        'Invoice': 'nunique',  # Frequency
        'InvoiceDate': 'max'   # Last purchase for Recency
    }).reset_index()
    
    # Rename columns
    rfm_df.columns = ['CustomerID', 'MonetaryValue', 'Frequency', 'LastPurchaseDate']
    
    # Calculate Recency
    max_date = rfm_df['LastPurchaseDate'].max()
    rfm_df['Recency'] = (max_date - rfm_df['LastPurchaseDate']).dt.days
    
    print(f"✅ Created RFM features for {len(rfm_df)} customers")
    print("\n📊 RFM Sample:")
    print(rfm_df.head())
    
    print("\n📊 RFM Statistics:")
    print(rfm_df[['MonetaryValue', 'Frequency', 'Recency']].describe())

# ============================================
# STEP 6: OUTLIER DETECTION
# ============================================
print("\n" + "="*60)
print("STEP 6: OUTLIER DETECTION")
print("="*60)

# Create boxplots for outlier visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

sns.boxplot(y=rfm_df['MonetaryValue'], ax=axes[0], color='skyblue')
axes[0].set_title('Monetary Value Distribution')
axes[0].set_ylabel('Monetary Value')

sns.boxplot(y=rfm_df['Frequency'], ax=axes[1], color='lightgreen')
axes[1].set_title('Frequency Distribution')
axes[1].set_ylabel('Frequency')

sns.boxplot(y=rfm_df['Recency'], ax=axes[2], color='salmon')
axes[2].set_title('Recency Distribution')
axes[2].set_ylabel('Recency (Days)')

plt.tight_layout()
plt.savefig('1_outlier_detection.png')
plt.show()
print("✅ Outlier detection plot saved as '1_outlier_detection.png'")

# Function to remove outliers using IQR method
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Remove outliers from each feature
rfm_clean = rfm_df.copy()
outlier_counts = {}
for col in ['MonetaryValue', 'Frequency', 'Recency']:
    before_count = len(rfm_clean)
    rfm_clean = remove_outliers_iqr(rfm_clean, col)
    after_count = len(rfm_clean)
    outlier_counts[col] = before_count - after_count

print(f"\n📊 Outlier Removal Summary:")
for col, count in outlier_counts.items():
    print(f"   {col}: removed {count} outliers")

print(f"\n✅ Total customers after outlier removal: {len(rfm_clean)}")
print(f"   Original customers: {len(rfm_df)}")
print(f"   Removed: {len(rfm_df) - len(rfm_clean)} customers ({((len(rfm_df) - len(rfm_clean))/len(rfm_df)*100):.1f}%)")

# ============================================
# STEP 7: DATA NORMALIZATION
# ============================================
print("\n" + "="*60)
print("STEP 7: DATA NORMALIZATION")
print("="*60)

# Select features for clustering
features = ['MonetaryValue', 'Frequency', 'Recency']
X = rfm_clean[features]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("✅ Features standardized")
print(f"   Feature means after scaling: {X_scaled.mean(axis=0).round(2)}")
print(f"   Feature stds after scaling: {X_scaled.std(axis=0).round(2)}")

# ============================================
# STEP 8: FIND OPTIMAL NUMBER OF CLUSTERS
# ============================================
print("\n" + "="*60)
print("STEP 8: FINDING OPTIMAL CLUSTERS")
print("="*60)

# Test different k values
max_k = 10
inertia = []
silhouette_scores = []
k_values = range(2, max_k + 1)

print("\n📊 Testing different cluster counts:")
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    inertia.append(kmeans.inertia_)
    sil_score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(sil_score)
    
    print(f"   k={k}: Inertia={kmeans.inertia_:.0f}, Silhouette={sil_score:.4f}")

# Find optimal k
optimal_k = k_values[np.argmax(silhouette_scores)]
print(f"\n✅ Optimal k based on silhouette score: {optimal_k}")

# Plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Elbow curve
ax1.plot(k_values, inertia, 'bo-')
ax1.set_xlabel('Number of Clusters (k)')
ax1.set_ylabel('Inertia')
ax1.set_title('Elbow Method for Optimal k')
ax1.grid(True, alpha=0.3)
ax1.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7)

# Silhouette scores
ax2.plot(k_values, silhouette_scores, 'ro-')
ax2.set_xlabel('Number of Clusters (k)')
ax2.set_ylabel('Silhouette Score')
ax2.set_title('Silhouette Score Method')
ax2.grid(True, alpha=0.3)
ax2.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('2_optimal_k_selection.png')
plt.show()
print("✅ Optimal k plot saved as '2_optimal_k_selection.png'")

# ============================================
# STEP 9: APPLY K-MEANS CLUSTERING
# ============================================
print("\n" + "="*60)
print("STEP 9: APPLYING K-MEANS CLUSTERING")
print("="*60)

# Use optimal k (or set to 4 if optimal_k is too high/low)
n_clusters = min(optimal_k, 5)  # Cap at 5 for interpretability
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
rfm_clean['Cluster'] = kmeans.fit_predict(X_scaled)

print(f"✅ Clustering completed with {n_clusters} clusters")
print("\n📊 Cluster Distribution:")
cluster_dist = rfm_clean['Cluster'].value_counts().sort_index()
for cluster, count in cluster_dist.items():
    percentage = (count / len(rfm_clean)) * 100
    print(f"   Cluster {cluster}: {count} customers ({percentage:.1f}%)")

# Calculate cluster profiles
print("\n📊 Cluster Profiles (Average Values):")
cluster_profiles = rfm_clean.groupby('Cluster')[features].mean().round(2)
print(cluster_profiles)

# ============================================
# STEP 10: VISUALIZE CLUSTERS
# ============================================
print("\n" + "="*60)
print("STEP 10: VISUALIZING CLUSTERS")
print("="*60)

# 3D Visualization
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Create color map
colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown'][:n_clusters]

for cluster in range(n_clusters):
    cluster_data = rfm_clean[rfm_clean['Cluster'] == cluster]
    ax.scatter(cluster_data['MonetaryValue'], 
               cluster_data['Frequency'], 
               cluster_data['Recency'],
               c=colors[cluster], 
               label=f'Cluster {cluster}', 
               s=50, alpha=0.6)

ax.set_xlabel('Monetary Value')
ax.set_ylabel('Frequency')
ax.set_zlabel('Recency (Days)')
ax.set_title('Customer Segments 3D Visualization')
ax.legend()

plt.savefig('3_customer_segments_3d.png')
plt.show()
print("✅ 3D visualization saved as '3_customer_segments_3d.png'")

# Pair plot
plt.figure(figsize=(12, 8))
pairplot = sns.pairplot(rfm_clean, vars=features, hue='Cluster', palette='Set1')
pairplot.fig.suptitle('Customer Segments Pair Plot', y=1.02)
plt.savefig('4_pairplot_clusters.png')
plt.show()
print("✅ Pair plot saved as '4_pairplot_clusters.png'")

# Cluster characteristics visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, feature in enumerate(features):
    sns.boxplot(x='Cluster', y=feature, data=rfm_clean, ax=axes[i], palette='Set2')
    axes[i].set_title(f'{feature} by Cluster')
    axes[i].set_xlabel('Cluster')

plt.tight_layout()
plt.savefig('5_cluster_characteristics.png')
plt.show()
print("✅ Cluster characteristics saved as '5_cluster_characteristics.png'")

# ============================================
# STEP 11: LABEL SEGMENTS
# ============================================
print("\n" + "="*60)
print("STEP 11: LABELING SEGMENTS")
print("="*60)

# Calculate global averages for comparison
global_avg_monetary = rfm_clean['MonetaryValue'].mean()
global_avg_frequency = rfm_clean['Frequency'].mean()
global_avg_recency = rfm_clean['Recency'].mean()

# Create segment labels based on characteristics
segment_labels = {}
segment_descriptions = {}

for cluster in range(n_clusters):
    cluster_data = rfm_clean[rfm_clean['Cluster'] == cluster]
    
    m_avg = cluster_data['MonetaryValue'].mean()
    f_avg = cluster_data['Frequency'].mean()
    r_avg = cluster_data['Recency'].mean()
    
    # Determine segment label based on RFM values
    if r_avg < 30 and f_avg > global_avg_frequency * 1.5 and m_avg > global_avg_monetary * 1.5:
        label = "🌟 VIP Customers"
        description = "Highest value, most frequent, recently active"
    elif r_avg < 60 and f_avg > global_avg_frequency and m_avg > global_avg_monetary:
        label = "💎 Loyal Customers"
        description = "Above average in all metrics, recently active"
    elif r_avg > 180:
        label = "⚠️ At-Risk Customers"
        description = "Haven't purchased in 6+ months"
    elif r_avg < 30 and f_avg < global_avg_frequency * 0.5:
        label = "🆕 New Customers"
        description = "Recent first-time or low-frequency buyers"
    elif m_avg < global_avg_monetary * 0.5:
        label = "💰 Low Spenders"
        description = "Below average monetary value"
    elif f_avg < 2:
        label = "📅 Occasional Shoppers"
        description = "Infrequent purchasers"
    else:
        label = f"📊 Segment {cluster}"
        description = "Mixed characteristics"
    
    segment_labels[cluster] = label
    segment_descriptions[cluster] = description

rfm_clean['Segment'] = rfm_clean['Cluster'].map(segment_labels)
rfm_clean['SegmentDescription'] = rfm_clean['Cluster'].map(segment_descriptions)

print("\n📊 Segment Distribution:")
segment_counts = rfm_clean['Segment'].value_counts()
for segment, count in segment_counts.items():
    percentage = (count / len(rfm_clean)) * 100
    cluster_num = rfm_clean[rfm_clean['Segment'] == segment]['Cluster'].iloc[0]
    print(f"   {segment}: {count} customers ({percentage:.1f}%) - {segment_descriptions[cluster_num]}")

# ============================================
# STEP 12: EXPORT RESULTS
# ============================================
print("\n" + "="*60)
print("STEP 12: EXPORTING RESULTS")
print("="*60)

# Create summary DataFrame
summary_df = rfm_clean.groupby('Segment').agg({
    'CustomerID': 'count',
    'MonetaryValue': ['mean', 'sum', 'median'],
    'Frequency': ['mean', 'median'],
    'Recency': ['mean', 'median']
}).round(2)

# Flatten column names
summary_df.columns = ['CustomerCount', 'AvgMonetary', 'TotalMonetary', 'MedianMonetary', 
                      'AvgFrequency', 'MedianFrequency', 'AvgRecency', 'MedianRecency']
summary_df = summary_df.reset_index()

# Calculate revenue share
summary_df['RevenueShare(%)'] = (summary_df['TotalMonetary'] / summary_df['TotalMonetary'].sum() * 100).round(1)

# Sort by TotalMonetary descending
summary_df = summary_df.sort_values('TotalMonetary', ascending=False)

print("\n📊 Segment Summary:")
print(summary_df.to_string(index=False))

# Save main results to your specified file
output_file = 'customer_segmentation.csv'
rfm_clean.to_csv(output_file, index=False)
print(f"\n✅ Main results saved to '{output_file}'")

# Save summary to a separate file
summary_file = 'segment_summary.csv'
summary_df.to_csv(summary_file, index=False)
print(f"✅ Summary saved to '{summary_file}'")

# Verify the file was saved
if os.path.exists(output_file):
    file_size = os.path.getsize(output_file)
    print(f"✅ Verified: '{output_file}' saved successfully (Size: {file_size:,} bytes)")
    
    # Show first few rows of the saved file
    result_df = pd.read_csv(output_file)
    print(f"\n📊 First 5 rows of '{output_file}':")
    print(result_df[['CustomerID', 'MonetaryValue', 'Frequency', 'Recency', 'Cluster', 'Segment']].head())

# ============================================
# STEP 13: FINAL VISUALIZATION DASHBOARD
# ============================================
print("\n" + "="*60)
print("STEP 13: CREATING FINAL DASHBOARD")
print("="*60)

# Create a comprehensive dashboard
fig = plt.figure(figsize=(20, 12))

# 1. Segment distribution pie chart
ax1 = plt.subplot(2, 3, 1)
colors = plt.cm.Set3(np.linspace(0, 1, len(segment_counts)))
wedges, texts, autotexts = plt.pie(segment_counts.values, 
                                    labels=segment_counts.index, 
                                    autopct='%1.1f%%', 
                                    colors=colors, 
                                    startangle=90)
plt.title('Customer Segment Distribution', fontsize=14, fontweight='bold')

# 2. Monetary value by segment
ax2 = plt.subplot(2, 3, 2)
segment_order = summary_df['Segment'].values
monetary_means = [rfm_clean[rfm_clean['Segment'] == seg]['MonetaryValue'].mean() for seg in segment_order]
bars = plt.bar(range(len(segment_order)), monetary_means, color=colors)
plt.xticks(range(len(segment_order)), [s.split()[-1] for s in segment_order], rotation=45, ha='right')
plt.title('Average Monetary Value by Segment', fontsize=14, fontweight='bold')
plt.ylabel('Avg Monetary Value ($)')

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars, monetary_means)):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
             f'${val:,.0f}', ha='center', va='bottom', fontsize=9)

# 3. Frequency by segment
ax3 = plt.subplot(2, 3, 3)
frequency_means = [rfm_clean[rfm_clean['Segment'] == seg]['Frequency'].mean() for seg in segment_order]
bars = plt.bar(range(len(segment_order)), frequency_means, color=colors)
plt.xticks(range(len(segment_order)), [s.split()[-1] for s in segment_order], rotation=45, ha='right')
plt.title('Average Purchase Frequency by Segment', fontsize=14, fontweight='bold')
plt.ylabel('Avg Frequency')

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars, frequency_means)):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
             f'{val:.1f}', ha='center', va='bottom', fontsize=9)

# 4. Recency by segment
ax4 = plt.subplot(2, 3, 4)
recency_means = [rfm_clean[rfm_clean['Segment'] == seg]['Recency'].mean() for seg in segment_order]
bars = plt.bar(range(len(segment_order)), recency_means, color=colors)
plt.xticks(range(len(segment_order)), [s.split()[-1] for s in segment_order], rotation=45, ha='right')
plt.title('Average Recency by Segment', fontsize=14, fontweight='bold')
plt.ylabel('Avg Recency (Days)')

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars, recency_means)):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
             f'{val:.0f} days', ha='center', va='bottom', fontsize=9)

# 5. Revenue share pie chart
ax5 = plt.subplot(2, 3, 5)
revenue_share = rfm_clean.groupby('Segment')['MonetaryValue'].sum().sort_values(ascending=False)
plt.pie(revenue_share.values, labels=revenue_share.index, autopct='%1.1f%%',
        colors=plt.cm.Set2(np.linspace(0, 1, len(revenue_share))))
plt.title('Revenue Share by Segment', fontsize=14, fontweight='bold')

# 6. Customer count by segment with revenue info
ax6 = plt.subplot(2, 3, 6)
customer_counts = [rfm_clean[rfm_clean['Segment'] == seg]['CustomerID'].count() for seg in segment_order]
bars = plt.barh(range(len(segment_order)), customer_counts, color=colors)
plt.yticks(range(len(segment_order)), [s.split()[-1] for s in segment_order])
plt.title('Customer Count by Segment', fontsize=14, fontweight='bold')
plt.xlabel('Number of Customers')

# Add value labels on bars
for i, (bar, count) in enumerate(zip(bars, customer_counts)):
    plt.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2, 
             f'{count:,}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('6_segmentation_dashboard.png', dpi=100, bbox_inches='tight')
plt.show()
print("✅ Dashboard saved as '6_segmentation_dashboard.png'")

# ============================================
# STEP 14: ADDITIONAL INSIGHTS
# ============================================
print("\n" + "="*60)
print("STEP 14: KEY INSIGHTS")
print("="*60)

print("\n📊 Top 3 Segments by Revenue:")
top_3_revenue = summary_df.nlargest(3, 'TotalMonetary')[['Segment', 'TotalMonetary', 'RevenueShare(%)']]
for idx, row in top_3_revenue.iterrows():
    print(f"   {row['Segment']}: ${row['TotalMonetary']:,.2f} ({row['RevenueShare(%)']}% of total)")

print("\n📊 Segments with Highest Customer Count:")
top_3_customers = summary_df.nlargest(3, 'CustomerCount')[['Segment', 'CustomerCount', 'AvgMonetary']]
for idx, row in top_3_customers.iterrows():
    print(f"   {row['Segment']}: {row['CustomerCount']:,} customers (Avg spend: ${row['AvgMonetary']:,.2f})")

print("\n📊 Segments Needing Attention:")
at_risk = rfm_clean[rfm_clean['Segment'].str.contains('At-Risk', na=False)]
if not at_risk.empty:
    print(f"   At-Risk Customers: {len(at_risk)} customers")
    print(f"   Average days since last purchase: {at_risk['Recency'].mean():.0f} days")
    print(f"   Potential revenue at risk: ${at_risk['MonetaryValue'].sum():,.2f}")

# ============================================
# STEP 15: FINAL SUMMARY
# ============================================
print("\n" + "="*60)
print("🎉 CUSTOMER SEGMENTATION COMPLETED SUCCESSFULLY!")
print("="*60)
print("\n📁 Files generated in current directory:")
print("   1. 1_outlier_detection.png - Outlier analysis")
print("   2. 2_optimal_k_selection.png - Optimal cluster selection")
print("   3. 3_customer_segments_3d.png - 3D segment visualization")
print("   4. 4_pairplot_clusters.png - Feature relationship analysis")
print("   5. 5_cluster_characteristics.png - Cluster characteristics")
print("   6. 6_segmentation_dashboard.png - Complete business dashboard")
print(f"   7. {output_file} - Complete customer segmentation data")
print("   8. segment_summary.csv - Segment-wise summary statistics")

print(f"\n📊 Total customers segmented: {len(rfm_clean):,}")
print(f"   Number of segments created: {n_clusters}")
print(f"   Average monetary value: ${rfm_clean['MonetaryValue'].mean():,.2f}")
print(f"   Average frequency: {rfm_clean['Frequency'].mean():.1f} purchases")
print(f"   Average recency: {rfm_clean['Recency'].mean():.0f} days")

print("\n💡 Next Steps:")
print("   1. Open 'customer_segmentation.csv' in Excel to view individual customer segments")
print("   2. Review 'segment_summary.csv' for marketing strategy decisions")
print("   3. Use '6_segmentation_dashboard.png' for presentations")
print("   4. Target VIP customers with loyalty programs")
print("   5. Re-engage at-risk customers with special offers")
print("   6. Nurture new customers to increase frequency")

print("\n" + "="*60)