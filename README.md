# AnalyticsLab_Project

Project title: CMA CGM 2 - No-Action Prediction Model

#### Team Members:  
- Guillaume Bonheure
- Andrea Zanon
- Max Petruzzi
- Victor Radermecker

This project was performed by four MIT Masters of Business Analytics students for the course 15.572 Analytics Lab. The project was done in collaboration with CMA CGM, the world's third-largest shipping company.

#### Goal:  
  The goal of the project was to help improve operational efficiency in customer support by reducing the burden of no-action emails. These are emails that are received but require no action on behalf of CMA CGM. Because these types of emails can take a variety of forms, including emails that require action from another party but not CMA CGM, this problem requires intelligent NLP and ML methodologies to ensure accuracy. Further, an actionable email that is incorrectly marked as no-action can be highly detrimental because these emails would be automatically removed from the inbox, so CMA CGM would never see these important communications. In this way, the recall of the model was prioritized to limit missed actionable emails.

#### Dataset:  
  The dataset consisted of 250k emails, each in the form of an HTML string of up to 60k characters. Each HTML string represented an entire email chain, so HTML parsers were used to format the emails without the HTML tags, and logic was introduced to mark where separate emails appear in each thread. Each prediction was made on the last (most recent) email in the thread, and the entire thread was utilized for making the decision.

#### Methodology:  
  A stacking methodology was utilized, meaning that the outputs of several models were used as predictors for a downstream XGBoost model. The first level included NLP models such as BERT fine-tuned for an initial "guess" in classifying the text body, DistilBERT for sentiment analysis, and Term Frequency - Inverse Document Frequency (TD-IDF) on the email content, team name, and email subject. Further, additional metadata features were engineered including the number of emails in the chain, whether CMA CGM was included in the "to" or "cc" line, and the length of the email. Lastly, these metadata features and first-level model outputs were all inputted into the downstream XGBoost model for final classification.

#### Results:  
  This work resulted in a **30% reduction** in no-action support tickets **(144k tickets per year)**, **1,872** freed up employee hours per year, and
an improved average response time by **22 seconds per ticket**.
