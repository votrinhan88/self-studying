# Abstraction: hide unnecessary details from users

class Email:
    def __connect(self):
        print('Connected to server.')

    def __prepare(self):
        print('Prepared email.')

    def __send(self):
        print('Sent email.')

    # This is a PUBLIC method
    def sendEmail(self):
        # These are PRIVATE methods
        self.__connect()
        self.__prepare()
        self.__send()

email = Email()
email.sendEmail()