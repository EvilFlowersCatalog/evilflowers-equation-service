from kafka.consumer import DefaultConsumer
from config.consumer_config import consumer_conf, consumer_subscriptions
from equationHandler.EquationService import EquationHandler
import json

class MyConsumer(DefaultConsumer):
    def msg_process(self, msg):
        
        json_string = msg.value().decode('utf-8')
        json_object = json.loads(json_string)
        
        equation_handler = EquationHandler(json_object["document_path"])
        latex_equations, python_equations = equation_handler.extract_equations_from_image()
        
        print("latex_equations", latex_equations)
        print("python_equations", python_equations)

if __name__ == "__main__":
    consumer = MyConsumer(consumer_conf, consumer_subscriptions)
    consumer.start_consume()
    print("Consumer started")
