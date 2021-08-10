from src.graph import State

MILK = 'Milk'
SUGAR = 'Sugar'
TEA = 'Tea'
COFFEE = 'Coffee'
HOT_CHOCOLATE = 'Hot chocolate'


class CoffeeMachine:
    def __init__(self):
        self.order = []

    def add_order(self, item):
        self.order.append(item)

    def reset_order(self):
        self.order = []

    def get_order(self):
        return self.order


coffee_machine = CoffeeMachine()


class MakeCoffee(State):

    def on_entry(self):
        coffee_machine.add_order(COFFEE)


class MakeTee(State):

    def on_entry(self):
        coffee_machine.add_order(TEA)


class MakeChocolate(State):

    def on_entry(self):
        coffee_machine.add_order(HOT_CHOCOLATE)


class AddMilk(State):

    def on_entry(self):
        coffee_machine.add_order(MILK)


class AddSugar(State):

    def on_entry(self):
        coffee_machine.add_order(SUGAR)


class VerifyOrder(State):

    def on_entry(self):
        print(coffee_machine.get_order())


class GiveOptions(State):

    def on_entry(self):
        coffee_machine.reset_order()
