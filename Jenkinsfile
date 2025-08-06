pipeline {
    agent any

    stages {
        stage('Install Dependencies') {
            steps {
                sh '''
                /opt/anaconda3/bin/python3 -m pip install --upgrade pip
                /opt/anaconda3/bin/python3 -m pip install pandas requests textblob transformers sentence-transformers scikit-learn numpy streamlit
                '''
            }
        }

        stage('Print LLM CSV') {
            steps {
                sh 'JENKINS_MODE=true /opt/anaconda3/bin/python3 llm_eval_local.py'
            }
        }
    }
}
