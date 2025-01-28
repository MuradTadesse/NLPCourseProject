class InferenceEngine:
    def __init__(self, model_server, response_templates):
        self.model_server = model_server
        self.response_templates = response_templates
        
    def get_response(self, query):
        try:
            response = self.model_server.predict(query)
            return response
        except Exception as e:
            print(f"Error in inference: {str(e)}")
            return "I apologize, I'm having trouble processing your request."