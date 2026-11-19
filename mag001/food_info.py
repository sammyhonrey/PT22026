# Each entry maps a Food-101 class name (the same names used in the model's real_class_mapping.json)  
# Categories not present here are handled by the application ("Nutrition information not available").

from __future__ import annotations

from typing import Dict, List, TypedDict


class FoodInfoEntry(TypedDict): # what every dictonary will have
    calories_per_100g: int
    ingredients: List[str]
    description: str

# the dictoraries
FOOD_INFO: Dict[str, FoodInfoEntry] = {
    "apple_pie": {
        "calories_per_100g": 237,
        "ingredients": [
            "apples",
            "flour",
            "butter",
            "sugar",
            "cinnamon",
            "nutmeg",
            "salt",
        ],
        "description": "Apple pie is a classic dessert made with a buttery pastry crust and a spiced apple filling. "
        "The apples are usually tossed with sugar and warm spices before baking until tender and golden. "
        "It is often served warm with whipped cream or vanilla ice cream.",
    },
    "baby_back_ribs": {
        "calories_per_100g": 320,
        "ingredients": [
            "pork ribs",
            "brown sugar",
            "paprika",
            "garlic powder",
            "black pepper",
            "barbecue sauce",
        ],
        "description": "Baby back ribs are a pork cut taken from the upper rib area near the spine. "
        "They are slow-cooked or smoked until tender, then glazed with a sweet and smoky barbecue sauce. "
        "The result is juicy meat with a caramelised, sticky exterior.",
    },
    "baklava": {
        "calories_per_100g": 428,
        "ingredients": [
            "phyllo pastry",
            "pistachios",
            "walnuts",
            "butter",
            "honey",
            "sugar",
            "lemon",
        ],
        "description": "Baklava is a rich Middle Eastern and Mediterranean pastry made from layers of thin phyllo dough. "
        "The layers are filled with chopped nuts, baked until crisp, and soaked in honey or sugar syrup. "
        "It is sweet, flaky, and typically cut into diamonds or squares.",
    },
    "beef_carpaccio": {
        "calories_per_100g": 160,
        "ingredients": [
            "beef tenderloin",
            "olive oil",
            "lemon juice",
            "parmesan",
            "capers",
            "arugula",
            "black pepper",
        ],
        "description": "Beef carpaccio consists of paper-thin slices of raw beef tenderloin. "
        "It is dressed with olive oil and lemon juice and topped with shaved parmesan, capers, and arugula. "
        "The dish is light, delicate, and served as an appetiser.",
    },
    "beef_tartare": {
        "calories_per_100g": 180,
        "ingredients": [
            "raw beef",
            "egg yolk",
            "capers",
            "shallots",
            "Dijon mustard",
            "Worcestershire sauce",
            "parsley",
        ],
        "description": "Beef tartare is a preparation of finely chopped raw beef mixed with seasonings and condiments. "
        "Common additions include capers, shallots, mustard, and a raw egg yolk. "
        "It is typically shaped into a patty and served with toast or crackers.",
    },
    "beet_salad": {
        "calories_per_100g": 120,
        "ingredients": [
            "beets",
            "goat cheese",
            "walnuts",
            "arugula",
            "olive oil",
            "balsamic vinegar",
        ],
        "description": "Beet salad combines roasted or boiled beets with fresh greens and a tangy dressing. "
        "It often includes goat cheese, walnuts, and arugula for contrasting textures and flavours. "
        "The dish is earthy, sweet, and visually vibrant.",
    },
    "beignets": {
        "calories_per_100g": 350,
        "ingredients": [
            "flour",
            "yeast",
            "milk",
            "butter",
            "eggs",
            "sugar",
            "powdered sugar",
        ],
        "description": "Beignets are square or rectangular fried dough pastries associated with New Orleans. "
        "They are deep-fried until puffy and golden, then generously dusted with powdered sugar. "
        "The texture is crisp outside and light and airy inside.",
    },
    "bibimbap": {
        "calories_per_100g": 130,
        "ingredients": [
            "rice",
            "beef",
            "spinach",
            "carrots",
            "bean sprouts",
            "mushrooms",
            "egg",
            "gochujang",
        ],
        "description": "Bibimbap is a Korean rice bowl topped with seasoned vegetables, meat, and a fried egg. "
        "The ingredients are arranged around the rice and mixed together with spicy gochujang sauce before eating. "
        "It is colourful, savoury, and often served in a hot stone bowl.",
    },
    "bread_pudding": {
        "calories_per_100g": 250,
        "ingredients": [
            "stale bread",
            "milk",
            "eggs",
            "sugar",
            "butter",
            "vanilla",
            "raisins",
            "cinnamon",
        ],
        "description": "Bread pudding is a baked dessert that uses stale bread soaked in a sweet custard. "
        "The mixture is flavoured with vanilla, cinnamon, and often raisins, then baked until set. "
        "It is served warm and may be finished with a sauce or cream.",
    },
    "breakfast_burrito": {
        "calories_per_100g": 210,
        "ingredients": [
            "flour tortilla",
            "scrambled eggs",
            "cheese",
            "bacon",
            "potatoes",
            "salsa",
        ],
        "description": "A breakfast burrito wraps scrambled eggs, cheese, and breakfast meats in a flour tortilla. "
        "It often includes potatoes, beans, or salsa for extra flavour and texture. "
        "It is a portable, hearty breakfast popular in the American Southwest.",
    },
    "bruschetta": {
        "calories_per_100g": 150,
        "ingredients": [
            "bread",
            "tomatoes",
            "garlic",
            "basil",
            "olive oil",
            "salt",
        ],
        "description": "Bruschetta is an Italian appetiser of grilled bread rubbed with garlic. "
        "It is topped with diced tomatoes, fresh basil, olive oil, and salt. "
        "The contrast of crisp toast and juicy tomatoes makes it a popular starter.",
    },
    "caesar_salad": {
        "calories_per_100g": 190,
        "ingredients": [
            "romaine lettuce",
            "parmesan",
            "croutons",
            "egg yolk",
            "olive oil",
            "lemon juice",
            "garlic",
            "anchovies",
        ],
        "description": "Caesar salad is a green salad made with crisp romaine lettuce and a creamy dressing. "
        "The dressing typically contains lemon, garlic, parmesan, egg yolk, and anchovies, and the salad is finished with croutons. "
        "It is rich, tangy, and one of the most widely recognised salads.",
    },
    "cannoli": {
        "calories_per_100g": 340,
        "ingredients": [
            "flour",
            "ricotta",
            "sugar",
            "chocolate chips",
            "pistachios",
            "cinnamon",
        ],
        "description": "Cannoli are Sicilian pastries consisting of fried tube-shaped shells filled with sweetened ricotta. "
        "The filling is often flavoured with vanilla, cinnamon, and chocolate chips or pistachios. "
        "They are crisp, creamy, and commonly served as a dessert.",
    },
    "caprese_salad": {
        "calories_per_100g": 140,
        "ingredients": [
            "tomatoes",
            "fresh mozzarella",
            "basil",
            "olive oil",
            "salt",
            "black pepper",
        ],
        "description": "Caprese salad is a simple Italian dish of sliced tomatoes, fresh mozzarella, and basil. "
        "It is dressed with olive oil, salt, and pepper, and sometimes a splash of balsamic vinegar. "
        "The colours mirror the Italian flag and the flavour relies on ripe, high-quality ingredients.",
    },
    "carrot_cake": {
        "calories_per_100g": 415,
        "ingredients": [
            "flour",
            "carrots",
            "sugar",
            "eggs",
            "vegetable oil",
            "cinnamon",
            "walnuts",
            "cream cheese",
        ],
        "description": "Carrot cake is a moist spice cake made with grated carrots, cinnamon, and often walnuts or raisins. "
        "It is typically layered and topped with a tangy cream cheese frosting. "
        "The cake is sweet, warmly spiced, and popular in many countries.",
    },
    "ceviche": {
        "calories_per_100g": 110,
        "ingredients": [
            "fresh fish",
            "lime juice",
            "red onion",
            "cilantro",
            "chili",
            "tomato",
        ],
        "description": "Ceviche is a Latin American dish of raw fish cured in citrus juice. "
        "The acid denatures the fish, giving it a firm texture and bright flavour. "
        "It is mixed with onion, cilantro, chili, and sometimes tomato or avocado.",
    },
    "cheese_plate": {
        "calories_per_100g": 350,
        "ingredients": [
            "assorted cheeses",
            "crackers",
            "grapes",
            "nuts",
            "olive oil",
            "honey",
        ],
        "description": "A cheese plate is a selection of cheeses served with accompaniments such as crackers, fruit, and nuts. "
        "The cheeses may range from soft and creamy to hard and aged. "
        "It is often served as an appetiser or a light meal with wine.",
    },
    "cheesecake": {
        "calories_per_100g": 321,
        "ingredients": [
            "cream cheese",
            "sugar",
            "eggs",
            "graham crackers",
            "butter",
            "vanilla",
        ],
        "description": "Cheesecake is a rich dessert with a creamy cheese-based filling on a crumb crust. "
        "The filling is usually made from cream cheese, sugar, eggs, and vanilla, and is baked or chilled until set. "
        "It can be served plain or topped with fruit, chocolate, or caramel.",
    },
    "chicken_curry": {
        "calories_per_100g": 180,
        "ingredients": [
            "chicken",
            "onion",
            "tomato",
            "garlic",
            "ginger",
            "curry spices",
            "coconut milk",
        ],
        "description": "Chicken curry is a spiced stew of chicken simmered in a sauce of onions, tomatoes, and aromatics. "
        "The flavour profile varies by region, from creamy coconut-based curries to dry, intensely spiced preparations. "
        "It is commonly served with rice, bread, or roti.",
    },
    "chicken_quesadilla": {
        "calories_per_100g": 250,
        "ingredients": [
            "flour tortilla",
            "chicken",
            "cheese",
            "bell peppers",
            "onion",
            "salsa",
        ],
        "description": "A chicken quesadilla is a folded tortilla filled with chicken and melted cheese. "
        "It is often grilled or pan-fried until the outside is crisp and the cheese is gooey. "
        "It is served with salsa, sour cream, or guacamole.",
    },
    "chicken_wings": {
        "calories_per_100g": 290,
        "ingredients": [
            "chicken wings",
            "flour",
            "hot sauce",
            "butter",
            "garlic powder",
            "paprika",
        ],
        "description": "Chicken wings are typically deep-fried or baked until crisp and then coated in sauce. "
        "Buffalo-style wings are tossed in a mixture of hot sauce and butter, but many other seasonings are common. "
        "They are a popular appetiser, especially with celery and blue cheese dip.",
    },
    "chocolate_cake": {
        "calories_per_100g": 371,
        "ingredients": [
            "flour",
            "cocoa powder",
            "sugar",
            "eggs",
            "butter",
            "milk",
            "chocolate",
        ],
        "description": "Chocolate cake is a baked dessert made with cocoa or melted chocolate in the batter. "
        "It can be light and fluffy or dense and fudgy, and is often layered with chocolate frosting. "
        "It is a celebration favourite around the world.",
    },
    "chocolate_mousse": {
        "calories_per_100g": 225,
        "ingredients": [
            "dark chocolate",
            "cream",
            "eggs",
            "sugar",
            "butter",
            "vanilla",
        ],
        "description": "Chocolate mousse is a light, airy dessert made by folding whipped cream or egg whites into melted chocolate. "
        "It has a smooth, creamy texture and an intense chocolate flavour. "
        "It is usually chilled and served in small cups or glasses.",
    },
    "churros": {
        "calories_per_100g": 450,
        "ingredients": [
            "flour",
            "water",
            "butter",
            "sugar",
            "cinnamon",
            "chocolate sauce",
        ],
        "description": "Churros are fried dough sticks with a crisp exterior and soft interior. "
        "They are often rolled in cinnamon sugar and served with thick hot chocolate for dipping. "
        "They are a popular snack in Spain, Latin America, and many other regions.",
    },
    "clam_chowder": {
        "calories_per_100g": 90,
        "ingredients": [
            "clams",
            "potatoes",
            "onion",
            "celery",
            "milk",
            "cream",
            "butter",
            "flour",
        ],
        "description": "Clam chowder is a creamy soup made with clams, potatoes, onions, and celery. "
        "New England style is thickened with a roux and enriched with milk or cream, while Manhattan style uses a tomato base. "
        "It is hearty, savoury, and often served with crackers or bread.",
    },
    "club_sandwich": {
        "calories_per_100g": 250,
        "ingredients": [
            "toast",
            "turkey",
            "bacon",
            "lettuce",
            "tomato",
            "mayonnaise",
        ],
        "description": "A club sandwich is a layered sandwich made with toasted bread, poultry, bacon, lettuce, and tomato. "
        "It is typically cut into triangles and held together with toothpicks. "
        "It is a filling, classic diner-style meal.",
    },
    "crab_cakes": {
        "calories_per_100g": 220,
        "ingredients": [
            "crab meat",
            "breadcrumbs",
            "egg",
            "mayonnaise",
            "mustard",
            "parsley",
            "Old Bay seasoning",
        ],
        "description": "Crab cakes are patties made from flaked crab meat mixed with breadcrumbs, egg, and seasonings. "
        "They are pan-fried or baked until golden and crisp on the outside. "
        "They are often served with tartar sauce or a lemon wedge.",
    },
    "creme_brulee": {
        "calories_per_100g": 300,
        "ingredients": [
            "cream",
            "egg yolks",
            "sugar",
            "vanilla",
        ],
        "description": "Crème brûlée is a French custard dessert with a rich, silky texture. "
        "It is baked in a water bath and topped with sugar that is caramelised into a crisp layer. "
        "The contrast between the hard caramel top and the cool custard is its signature feature.",
    },
    "croque_madame": {
        "calories_per_100g": 280,
        "ingredients": [
            "bread",
            "ham",
            "cheese",
            "butter",
            "flour",
            "milk",
            "egg",
        ],
        "description": "Croque madame is a French grilled ham and cheese sandwich topped with a fried egg. "
        "It is often covered in béchamel sauce and grated cheese before being grilled until bubbling. "
        "It is a rich, savoury dish commonly eaten for lunch.",
    },
    "cup_cakes": {
        "calories_per_100g": 305,
        "ingredients": [
            "flour",
            "sugar",
            "butter",
            "eggs",
            "milk",
            "vanilla",
            "frosting",
        ],
        "description": "Cupcakes are small individual cakes baked in paper liners. "
        "They are usually topped with frosting, sprinkles, or other decorations. "
        "They come in many flavours and are popular for parties and celebrations.",
    },
    "deviled_eggs": {
        "calories_per_100g": 220,
        "ingredients": [
            "eggs",
            "mayonnaise",
            "mustard",
            "paprika",
            "pickle relish",
            "salt",
        ],
        "description": "Deviled eggs are hard-boiled eggs that have been halved and filled with a seasoned yolk mixture. "
        "The filling typically combines the yolks with mayonnaise, mustard, and paprika. "
        "They are a classic party appetiser and picnic food.",
    },
    "donuts": {
        "calories_per_100g": 452,
        "ingredients": [
            "flour",
            "sugar",
            "yeast",
            "milk",
            "butter",
            "eggs",
            "glaze",
        ],
        "description": "Donuts are sweet fried dough pastries, either ring-shaped or filled. "
        "They can be glazed, frosted, powdered, or coated in cinnamon sugar. "
        "They are commonly eaten for breakfast or as a snack with coffee.",
    },
    "dumplings": {
        "calories_per_100g": 210,
        "ingredients": [
            "flour",
            "pork",
            "cabbage",
            "ginger",
            "garlic",
            "soy sauce",
            "scallions",
        ],
        "description": "Dumplings are parcels of dough filled with meat, vegetables, or both. "
        "They can be boiled, steamed, pan-fried, or deep-fried depending on the style. "
        "They are a staple in Chinese and many other Asian cuisines.",
    },
    "edamame": {
        "calories_per_100g": 121,
        "ingredients": [
            "soybeans",
            "salt",
            "water",
        ],
        "description": "Edamame are young soybeans harvested before they harden. "
        "They are usually boiled or steamed and served in the pod with salt. "
        "They are a simple, protein-rich snack common in Japanese cuisine.",
    },
    "eggs_benedict": {
        "calories_per_100g": 230,
        "ingredients": [
            "English muffin",
            "poached egg",
            "Canadian bacon",
            "hollandaise sauce",
            "chives",
        ],
        "description": "Eggs Benedict is a breakfast dish of poached eggs and Canadian bacon on a toasted English muffin. "
        "It is covered with rich hollandaise sauce and often garnished with chives. "
        "It is a brunch classic known for its creamy, tangy flavour.",
    },
    "escargots": {
        "calories_per_100g": 90,
        "ingredients": [
            "snails",
            "butter",
            "garlic",
            "parsley",
            "shallots",
            "white wine",
        ],
        "description": "Escargots are edible land snails, commonly prepared in French cuisine. "
        "They are baked with garlic-parsley butter and served in their shells or small dishes. "
        "The texture is chewy and the flavour comes largely from the butter and aromatics.",
    },
    "falafel": {
        "calories_per_100g": 333,
        "ingredients": [
            "chickpeas",
            "onion",
            "garlic",
            "parsley",
            "cilantro",
            "cumin",
            "flour",
        ],
        "description": "Falafel are deep-fried patties made from ground chickpeas or fava beans. "
        "They are seasoned with herbs and spices such as parsley, cilantro, and cumin. "
        "They are commonly served in pita bread with salad and tahini sauce.",
    },
    "filet_mignon": {
        "calories_per_100g": 220,
        "ingredients": [
            "beef tenderloin",
            "butter",
            "garlic",
            "rosemary",
            "salt",
            "black pepper",
        ],
        "description": "Filet mignon is a tender cut of beef taken from the tenderloin. "
        "It is usually grilled or pan-seared and finished with butter, garlic, and herbs. "
        "It is lean, mild in flavour, and one of the most prized steak cuts.",
    },
    "fish_and_chips": {
        "calories_per_100g": 280,
        "ingredients": [
            "white fish",
            "flour",
            "beer",
            "potatoes",
            "oil",
            "salt",
            "malt vinegar",
        ],
        "description": "Fish and chips is a British dish of battered, deep-fried fish served with thick-cut fries. "
        "The batter is often made with flour and beer or water, and the dish is seasoned with salt and vinegar. "
        "It is an iconic takeaway meal commonly served with mushy peas or tartar sauce.",
    },
    "foie_gras": {
        "calories_per_100g": 462,
        "ingredients": [
            "duck liver",
            "goose liver",
            "salt",
            "pepper",
            "brandy",
        ],
        "description": "Foie gras is a rich French delicacy made from the fattened liver of ducks or geese. "
        "It is typically seared or served as a terrine and has a very smooth, buttery texture. "
        "It is controversial because of the force-feeding involved in its production.",
    },
    "french_fries": {
        "calories_per_100g": 312,
        "ingredients": [
            "potatoes",
            "vegetable oil",
            "salt",
        ],
        "description": "French fries are strips of potato that are deep-fried until golden and crisp. "
        "They are usually salted and served as a side dish with burgers, sandwiches, or fried fish. "
        "They are one of the most common fast-food items worldwide.",
    },
    "french_onion_soup": {
        "calories_per_100g": 75,
        "ingredients": [
            "onions",
            "beef broth",
            "butter",
            "flour",
            "baguette",
            "gruyere cheese",
            "thyme",
        ],
        "description": "French onion soup is a savoury soup made with slowly caramelised onions and beef broth. "
        "It is traditionally topped with toasted baguette slices and melted Gruyère cheese. "
        "The soup is rich, deeply flavoured, and often baked until the cheese is bubbling.",
    },
    "french_toast": {
        "calories_per_100g": 229,
        "ingredients": [
            "bread",
            "eggs",
            "milk",
            "cinnamon",
            "vanilla",
            "butter",
            "maple syrup",
        ],
        "description": "French toast is made by soaking bread in a mixture of eggs and milk, then frying it until golden. "
        "It is often flavoured with cinnamon and vanilla and served with syrup, butter, or fruit. "
        "It is a popular sweet breakfast dish.",
    },
    "fried_calamari": {
        "calories_per_100g": 220,
        "ingredients": [
            "squid",
            "flour",
            "breadcrumbs",
            "egg",
            "oil",
            "lemon",
            "marinara sauce",
        ],
        "description": "Fried calamari consists of squid rings or tentacles coated in batter or breadcrumbs and deep-fried. "
        "It is served hot and crisp, usually with lemon wedges and a dipping sauce. "
        "It is a common appetiser in Mediterranean and seafood restaurants.",
    },
    "fried_rice": {
        "calories_per_100g": 168,
        "ingredients": [
            "rice",
            "eggs",
            "soy sauce",
            "scallions",
            "carrots",
            "peas",
            "vegetable oil",
        ],
        "description": "Fried rice is a dish of cooked rice stir-fried with vegetables, eggs, and often meat or seafood. "
        "It is seasoned with soy sauce and other condiments and cooked over high heat for a slightly smoky flavour. "
        "It is a versatile way to use leftover rice and is popular across Asia.",
    },
    "frozen_yogurt": {
        "calories_per_100g": 159,
        "ingredients": [
            "yogurt",
            "sugar",
            "milk",
            "fruit",
            "vanilla",
        ],
        "description": "Frozen yogurt is a frozen dessert made from yogurt, milk, and sugar. "
        "It has a tangy flavour and a lighter texture than ice cream. "
        "It is often served with fruit, nuts, or candy toppings.",
    },
    "garlic_bread": {
        "calories_per_100g": 350,
        "ingredients": [
            "bread",
            "butter",
            "garlic",
            "parsley",
            "salt",
        ],
        "description": "Garlic bread is toasted bread topped with garlic butter and herbs. "
        "It is often baked until the edges are crisp and the butter is melted. "
        "It is a common side dish for pasta, pizza, and soups.",
    },
    "gnocchi": {
        "calories_per_100g": 130,
        "ingredients": [
            "potatoes",
            "flour",
            "egg",
            "salt",
            "butter",
            "parmesan",
        ],
        "description": "Gnocchi are small Italian dumplings made from potato, flour, and sometimes egg. "
        "They are boiled until they float and then often tossed with butter, cheese, or tomato sauce. "
        "They have a soft, pillowy texture and are a comfort-food favourite.",
    },
    "greek_salad": {
        "calories_per_100g": 120,
        "ingredients": [
            "tomatoes",
            "cucumber",
            "red onion",
            "feta cheese",
            "olives",
            "olive oil",
            "oregano",
        ],
        "description": "Greek salad is a fresh salad of tomatoes, cucumber, red onion, and olives. "
        "It is topped with feta cheese, olive oil, and oregano, and contains no lettuce in the traditional version. "
        "It is light, tangy, and a staple of Greek cuisine.",
    },
    "grilled_cheese_sandwich": {
        "calories_per_100g": 300,
        "ingredients": [
            "bread",
            "cheese",
            "butter",
        ],
        "description": "A grilled cheese sandwich is made by heating cheese between slices of buttered bread until melted. "
        "The bread becomes crisp and golden while the cheese turns gooey. "
        "It is a simple, comforting meal often served with soup.",
    },
    "grilled_salmon": {
        "calories_per_100g": 206,
        "ingredients": [
            "salmon fillet",
            "olive oil",
            "lemon",
            "garlic",
            "salt",
            "black pepper",
            "dill",
        ],
        "description": "Grilled salmon is a fillet of salmon cooked over direct heat until the outside is charred and the inside is flaky. "
        "It is often seasoned with lemon, garlic, and herbs. "
        "It is a popular source of protein and omega-3 fatty acids.",
    },
    "guacamole": {
        "calories_per_100g": 160,
        "ingredients": [
            "avocado",
            "lime juice",
            "onion",
            "tomato",
            "cilantro",
            "jalapeño",
            "salt",
        ],
        "description": "Guacamole is a Mexican dip made from mashed avocado mixed with lime juice, onion, tomato, and cilantro. "
        "It is seasoned with salt and sometimes chili for heat. "
        "It is commonly served with tortilla chips, tacos, or grilled meats.",
    },
    "gyoza": {
        "calories_per_100g": 220,
        "ingredients": [
            "flour",
            "pork",
            "cabbage",
            "garlic",
            "ginger",
            "soy sauce",
            "sesame oil",
        ],
        "description": "Gyoza are Japanese pan-fried dumplings with a thin wheat wrapper and a savoury filling. "
        "They are typically filled with pork, cabbage, garlic, and ginger, then fried and steamed. "
        "They are served with a dipping sauce of soy sauce and vinegar.",
    },
    "hamburger": {
        "calories_per_100g": 295,
        "ingredients": [
            "ground beef",
            "burger bun",
            "lettuce",
            "tomato",
            "onion",
            "cheese",
            "pickles",
            "ketchup",
        ],
        "description": "A hamburger is a sandwich consisting of a cooked ground-beef patty in a bun. "
        "It is commonly topped with lettuce, tomato, onion, cheese, pickles, and condiments. "
        "It is one of the most iconic fast-food dishes in the world.",
    },
    "hot_and_sour_soup": {
        "calories_per_100g": 60,
        "ingredients": [
            "broth",
            "tofu",
            "mushrooms",
            "bamboo shoots",
            "eggs",
            "vinegar",
            "white pepper",
            "cornstarch",
        ],
        "description": "Hot and sour soup is a Chinese soup known for its balance of spicy and tangy flavours. "
        "It typically contains tofu, mushrooms, bamboo shoots, and egg ribbons in a thickened broth. "
        "White pepper provides the heat while vinegar supplies the sour note.",
    },
    "hot_dog": {
        "calories_per_100g": 290,
        "ingredients": [
            "sausage",
            "hot dog bun",
            "mustard",
            "ketchup",
            "onions",
            "pickle relish",
        ],
        "description": "A hot dog is a cooked sausage served in a sliced bun. "
        "It is commonly topped with mustard, ketchup, onions, and relish, though regional variations are endless. "
        "It is a popular street food and casual meal.",
    },
    "huevos_rancheros": {
        "calories_per_100g": 150,
        "ingredients": [
            "eggs",
            "corn tortillas",
            "tomato",
            "onion",
            "chili",
            "beans",
            "cheese",
            "cilantro",
        ],
        "description": "Huevos rancheros is a Mexican breakfast dish of fried eggs served on warm corn tortillas. "
        "The eggs are topped with a spicy tomato-chili sauce and often accompanied by beans and cheese. "
        "It is a hearty, savoury way to start the day.",
    },
    "hummus": {
        "calories_per_100g": 166,
        "ingredients": [
            "chickpeas",
            "tahini",
            "olive oil",
            "lemon juice",
            "garlic",
            "cumin",
            "salt",
        ],
        "description": "Hummus is a creamy Middle Eastern dip made from blended chickpeas, tahini, olive oil, and lemon juice. "
        "It is flavoured with garlic and cumin and often garnished with paprika and olive oil. "
        "It is served with pita bread, vegetables, or as part of a mezze spread.",
    },
    "ice_cream": {
        "calories_per_100g": 207,
        "ingredients": [
            "milk",
            "cream",
            "sugar",
            "egg yolks",
            "vanilla",
        ],
        "description": "Ice cream is a frozen dessert made from dairy, sugar, and flavourings. "
        "It is churned while freezing to incorporate air and create a smooth, creamy texture. "
        "It comes in countless flavours and is often served in cones, cups, or with toppings.",
    },
    "lasagna": {
        "calories_per_100g": 135,
        "ingredients": [
            "lasagna noodles",
            "ground beef",
            "tomato sauce",
            "ricotta",
            "mozzarella",
            "parmesan",
            "herbs",
        ],
        "description": "Lasagna is an Italian baked pasta dish made with layers of noodles, meat or vegetable sauce, and cheese. "
        "It is typically assembled with ricotta, mozzarella, and parmesan and baked until bubbly. "
        "It is a hearty, comforting meal often served at family gatherings.",
    },
    "lobster_bisque": {
        "calories_per_100g": 120,
        "ingredients": [
            "lobster",
            "cream",
            "butter",
            "onion",
            "carrot",
            "celery",
            "brandy",
            "tomato paste",
        ],
        "description": "Lobster bisque is a rich, creamy soup made from lobster shells and meat. "
        "The shells are simmered to build flavour, and the soup is finished with cream and sometimes brandy. "
        "It is a luxurious seafood starter with a deep orange colour.",
    },
    "lobster_roll_sandwich": {
        "calories_per_100g": 220,
        "ingredients": [
            "lobster meat",
            "mayonnaise",
            "celery",
            "lemon juice",
            "butter",
            "hot dog bun",
            "chives",
        ],
        "description": "A lobster roll is a New England sandwich of chilled lobster meat in a toasted, buttered bun. "
        "The lobster is lightly dressed with mayonnaise, lemon, and celery or served warm with butter. "
        "It is a summer seafood favourite, especially in coastal New England.",
    },
    "macaroni_and_cheese": {
        "calories_per_100g": 164,
        "ingredients": [
            "macaroni",
            "cheddar cheese",
            "milk",
            "butter",
            "flour",
            "mustard powder",
        ],
        "description": "Macaroni and cheese is a baked or stovetop dish of pasta coated in a creamy cheese sauce. "
        "It is often made with cheddar and a roux-based béchamel, then topped with breadcrumbs and baked. "
        "It is a classic comfort food in many English-speaking countries.",
    },
    "macarons": {
        "calories_per_100g": 407,
        "ingredients": [
            "almond flour",
            "egg whites",
            "sugar",
            "powdered sugar",
            "buttercream",
            "food colouring",
        ],
        "description": "Macarons are delicate French sandwich cookies made from almond flour, egg whites, and sugar. "
        "They have a crisp shell and a soft, chewy interior, usually filled with buttercream, ganache, or jam. "
        "They are prized for their smooth tops and ruffled edges.",
    },
    "miso_soup": {
        "calories_per_100g": 35,
        "ingredients": [
            "miso paste",
            "dashi",
            "tofu",
            "wakame seaweed",
            "scallions",
        ],
        "description": "Miso soup is a traditional Japanese soup made with fermented soybean paste dissolved in dashi stock. "
        "It commonly contains tofu, wakame seaweed, and scallions. "
        "It is light, savoury, and often served alongside sushi or rice dishes.",
    },
    "mussels": {
        "calories_per_100g": 172,
        "ingredients": [
            "mussels",
            "garlic",
            "white wine",
            "butter",
            "parsley",
            "shallots",
        ],
        "description": "Mussels are bivalve shellfish commonly steamed with garlic, white wine, and herbs. "
        "They are served in their shells with the cooking broth, often accompanied by bread or fries. "
        "They are a popular seafood dish in Belgium, France, and many coastal regions.",
    },
    "nachos": {
        "calories_per_100g": 340,
        "ingredients": [
            "tortilla chips",
            "cheddar cheese",
            "jalapeños",
            "beans",
            "sour cream",
            "guacamole",
            "salsa",
        ],
        "description": "Nachos are a Mexican-inspired snack of tortilla chips topped with melted cheese and other ingredients. "
        "Common toppings include jalapeños, beans, sour cream, guacamole, and salsa. "
        "They are typically shared as an appetiser or party food.",
    },
    "omelette": {
        "calories_per_100g": 154,
        "ingredients": [
            "eggs",
            "butter",
            "milk",
            "cheese",
            "ham",
            "vegetables",
            "salt",
        ],
        "description": "An omelette is a dish made by beating eggs and cooking them in a pan until set. "
        "It can be folded around cheese, vegetables, meat, or herbs. "
        "It is a quick, versatile meal served for breakfast, lunch, or dinner.",
    },
    "onion_rings": {
        "calories_per_100g": 411,
        "ingredients": [
            "onions",
            "flour",
            "breadcrumbs",
            "egg",
            "milk",
            "oil",
            "salt",
        ],
        "description": "Onion rings are slices of onion coated in batter or breadcrumbs and deep-fried. "
        "They are crisp, golden, and slightly sweet from the onion. "
        "They are commonly served as a side dish with burgers or as a snack.",
    },
    "oysters": {
        "calories_per_100g": 81,
        "ingredients": [
            "oysters",
            "lemon",
            "mignonette sauce",
            "hot sauce",
            "butter",
            "garlic",
        ],
        "description": "Oysters are bivalve molluscs often served raw on the half shell with lemon or mignonette sauce. "
        "They can also be grilled, fried, or baked with butter and garlic. "
        "They have a briny, mineral flavour and a soft texture.",
    },
    "pad_thai": {
        "calories_per_100g": 190,
        "ingredients": [
            "rice noodles",
            "shrimp",
            "egg",
            "tamarind",
            "fish sauce",
            "peanuts",
            "bean sprouts",
            "lime",
        ],
        "description": "Pad Thai is a Thai stir-fried noodle dish made with rice noodles, eggs, and a sweet-sour tamarind sauce. "
        "It typically includes shrimp or chicken, peanuts, bean sprouts, and lime. "
        "It is one of the most internationally recognised Thai dishes.",
    },
    "paella": {
        "calories_per_100g": 156,
        "ingredients": [
            "rice",
            "saffron",
            "shrimp",
            "mussels",
            "chicken",
            "bell peppers",
            "tomato",
            "olive oil",
        ],
        "description": "Paella is a Spanish rice dish originating in Valencia. "
        "It is cooked in a wide pan with saffron, seafood, chicken, and vegetables, allowing the rice to develop a crispy base. "
        "It is often served communally and is a symbol of Spanish cuisine.",
    },
    "pancakes": {
        "calories_per_100g": 227,
        "ingredients": [
            "flour",
            "milk",
            "eggs",
            "baking powder",
            "sugar",
            "butter",
            "maple syrup",
        ],
        "description": "Pancakes are flat, round cakes made from a batter of flour, milk, eggs, and baking powder. "
        "They are cooked on a griddle until golden and fluffy. "
        "They are commonly served with butter, syrup, fruit, or other toppings.",
    },
    "panna_cotta": {
        "calories_per_100g": 250,
        "ingredients": [
            "cream",
            "milk",
            "sugar",
            "gelatin",
            "vanilla",
        ],
        "description": "Panna cotta is an Italian dessert made from sweetened cream thickened with gelatin. "
        "It is chilled until set and has a smooth, silky texture. "
        "It is often served with fruit coulis, caramel, or chocolate sauce.",
    },
    "peking_duck": {
        "calories_per_100g": 337,
        "ingredients": [
            "duck",
            "maltose",
            "soy sauce",
            "ginger",
            "scallions",
            "cucumber",
            "pancakes",
            "hoisin sauce",
        ],
        "description": "Peking duck is a famous Chinese dish known for its crispy, lacquered skin. "
        "The duck is roasted and traditionally carved at the table, then served with thin pancakes, scallions, cucumber, and hoisin sauce. "
        "It is a celebratory dish with a long history in Beijing.",
    },
    "pho": {
        "calories_per_100g": 90,
        "ingredients": [
            "rice noodles",
            "beef broth",
            "beef",
            "ginger",
            "star anise",
            "cinnamon",
            "basil",
            "lime",
        ],
        "description": "Pho is a Vietnamese noodle soup with a clear, deeply flavoured broth. "
        "It is typically made with beef or chicken, rice noodles, herbs, and aromatics such as ginger and star anise. "
        "It is often garnished with basil, lime, and bean sprouts.",
    },
    "pizza": {
        "calories_per_100g": 266,
        "ingredients": [
            "pizza dough",
            "tomato sauce",
            "mozzarella",
            "olive oil",
            "basil",
            "oregano",
        ],
        "description": "Pizza is a baked flatbread topped with tomato sauce, cheese, and various ingredients. "
        "The Neapolitan style is known for its thin, blistered crust and simple toppings. "
        "It is one of the most popular foods in the world, with many regional variations.",
    },
    "pork_chop": {
        "calories_per_100g": 231,
        "ingredients": [
            "pork chop",
            "salt",
            "black pepper",
            "garlic",
            "butter",
            "rosemary",
            "thyme",
        ],
        "description": "A pork chop is a cut of pork taken from the loin or rib area. "
        "It is usually grilled, pan-fried, or baked and seasoned with salt, pepper, and herbs. "
        "It is a common main course served with potatoes or vegetables.",
    },
    "poutine": {
        "calories_per_100g": 230,
        "ingredients": [
            "french fries",
            "cheese curds",
            "gravy",
        ],
        "description": "Poutine is a Canadian dish of french fries topped with cheese curds and hot gravy. "
        "The gravy melts the curds, creating a rich, savoury combination. "
        "It originated in Quebec and is now a popular comfort food across Canada.",
    },
    "prime_rib": {
        "calories_per_100g": 300,
        "ingredients": [
            "beef prime rib",
            "salt",
            "black pepper",
            "garlic",
            "rosemary",
            "horseradish",
        ],
        "description": "Prime rib is a tender, marbled cut of beef roasted as a whole joint. "
        "It is seasoned simply with salt, pepper, and herbs and cooked until pink in the centre. "
        "It is often served with horseradish sauce and is a traditional holiday meal.",
    },
    "pulled_pork_sandwich": {
        "calories_per_100g": 240,
        "ingredients": [
            "pork shoulder",
            "barbecue sauce",
            "bun",
            "coleslaw",
            "pickles",
            "vinegar",
        ],
        "description": "A pulled pork sandwich is made with slow-cooked, shredded pork tossed in barbecue sauce and served in a bun. "
        "It is often topped with coleslaw or pickles for crunch and acidity. "
        "It is a staple of American barbecue culture.",
    },
    "ramen": {
        "calories_per_100g": 120,
        "ingredients": [
            "ramen noodles",
            "broth",
            "pork",
            "egg",
            "nori",
            "scallions",
            "bamboo shoots",
        ],
        "description": "Ramen is a Japanese noodle soup with wheat noodles served in a savoury broth. "
        "It is typically topped with sliced pork, a soft-boiled egg, nori, and scallions. "
        "There are many regional styles, including shoyu, miso, shio, and tonkotsu.",
    },
    "ravioli": {
        "calories_per_100g": 170,
        "ingredients": [
            "pasta dough",
            "ricotta",
            "spinach",
            "egg",
            "parmesan",
            "tomato sauce",
            "butter",
        ],
        "description": "Ravioli are Italian stuffed pasta parcels, usually square or round. "
        "They are filled with cheese, meat, or vegetables and served with sauce or butter. "
        "They are a classic comfort food and come in many regional variations.",
    },
    "red_velvet_cake": {
        "calories_per_100g": 350,
        "ingredients": [
            "flour",
            "cocoa powder",
            "buttermilk",
            "red food colouring",
            "butter",
            "sugar",
            "cream cheese frosting",
        ],
        "description": "Red velvet cake is a moist, mildly chocolate-flavoured cake with a distinctive red colour. "
        "It is typically layered and frosted with cream cheese icing. "
        "It is a popular celebration cake, especially in the United States.",
    },
    "risotto": {
        "calories_per_100g": 166,
        "ingredients": [
            "arborio rice",
            "broth",
            "onion",
            "white wine",
            "butter",
            "parmesan",
            "olive oil",
        ],
        "description": "Risotto is an Italian rice dish cooked slowly with broth until creamy. "
        "It is often finished with butter and parmesan and can include mushrooms, seafood, or vegetables. "
        "The grains should remain slightly firm to the bite.",
    },
    "samosa": {
        "calories_per_100g": 262,
        "ingredients": [
            "flour",
            "potatoes",
            "peas",
            "onion",
            "cumin",
            "coriander",
            "chili",
            "oil",
        ],
        "description": "Samosas are fried or baked pastries filled with spiced potatoes, peas, and sometimes meat. "
        "They are triangular in shape and have a crisp, flaky exterior. "
        "They are a popular snack in South Asia and the Middle East.",
    },
    "sashimi": {
        "calories_per_100g": 130,
        "ingredients": [
            "raw fish",
            "soy sauce",
            "wasabi",
            "pickled ginger",
            "daikon radish",
        ],
        "description": "Sashimi is thinly sliced raw fish or seafood served without rice. "
        "It is typically accompanied by soy sauce, wasabi, and pickled ginger. "
        "Freshness and knife skills are essential, and the dish highlights the natural flavour of the fish.",
    },
    "scallops": {
        "calories_per_100g": 111,
        "ingredients": [
            "scallops",
            "butter",
            "garlic",
            "lemon",
            "parsley",
            "salt",
            "black pepper",
        ],
        "description": "Scallops are bivalve molluscs with sweet, tender adductor muscle meat. "
        "They are often seared in butter until golden on the outside and served with lemon and herbs. "
        "They are a popular seafood appetiser or main course.",
    },
    "seaweed_salad": {
        "calories_per_100g": 70,
        "ingredients": [
            "wakame seaweed",
            "sesame oil",
            "soy sauce",
            "rice vinegar",
            "sugar",
            "sesame seeds",
        ],
        "description": "Seaweed salad is a Japanese side dish made from wakame seaweed. "
        "It is dressed with sesame oil, soy sauce, vinegar, and sugar, and sprinkled with sesame seeds. "
        "It is light, refreshing, and commonly served at sushi restaurants.",
    },
    "shrimp_and_grits": {
        "calories_per_100g": 220,
        "ingredients": [
            "shrimp",
            "grits",
            "butter",
            "cheese",
            "garlic",
            "bacon",
            "scallions",
        ],
        "description": "Shrimp and grits is a Southern American dish of creamy grits topped with sautéed shrimp. "
        "It is often flavoured with bacon, garlic, butter, and cheese, and finished with scallions. "
        "It is a savoury comfort food with roots in the coastal South.",
    },
    "spaghetti_bolognese": {
        "calories_per_100g": 150,
        "ingredients": [
            "spaghetti",
            "ground beef",
            "tomato sauce",
            "onion",
            "carrot",
            "celery",
            "red wine",
            "parmesan",
        ],
        "description": "Spaghetti bolognese is a pasta dish with a rich meat and tomato sauce. "
        "The sauce is slowly simmered with aromatics such as onion, carrot, and celery, and often a splash of wine. "
        "It is served over spaghetti and topped with parmesan.",
    },
    "spaghetti_carbonara": {
        "calories_per_100g": 190,
        "ingredients": [
            "spaghetti",
            "guanciale",
            "egg yolks",
            "pecorino romano",
            "black pepper",
        ],
        "description": "Spaghetti carbonara is a Roman pasta dish made with eggs, cheese, and cured pork. "
        "The heat of the pasta cooks the egg mixture into a creamy sauce without using cream. "
        "Black pepper and pecorino give it a sharp, savoury flavour.",
    },
    "spring_rolls": {
        "calories_per_100g": 154,
        "ingredients": [
            "spring roll wrappers",
            "cabbage",
            "carrots",
            "mushrooms",
            "bean sprouts",
            "soy sauce",
            "oil",
        ],
        "description": "Spring rolls are cylindrical rolls filled with vegetables and sometimes meat. "
        "They are deep-fried or served fresh with a dipping sauce. "
        "They are a popular appetiser across East and Southeast Asia.",
    },
    "steak": {
        "calories_per_100g": 271,
        "ingredients": [
            "beef steak",
            "salt",
            "black pepper",
            "butter",
            "garlic",
            "rosemary",
        ],
        "description": "Steak is a cut of beef cooked by grilling, pan-searing, or broiling. "
        "It is typically seasoned with salt and pepper and served with butter or a sauce. "
        "Doneness ranges from rare to well done, depending on preference.",
    },
    "strawberry_shortcake": {
        "calories_per_100g": 280,
        "ingredients": [
            "shortcake",
            "strawberries",
            "whipped cream",
            "sugar",
            "butter",
            "vanilla",
        ],
        "description": "Strawberry shortcake is a dessert made with layers of cake or biscuit, fresh strawberries, and whipped cream. "
        "The berries are often macerated with sugar to release their juices. "
        "It is a light, fruity dessert popular in summer.",
    },
    "sushi": {
        "calories_per_100g": 150,
        "ingredients": [
            "sushi rice",
            "nori",
            "fish",
            "seafood",
            "vinegar",
            "wasabi",
            "soy sauce",
        ],
        "description": "Sushi is a Japanese dish of vinegared rice combined with fish, seafood, or vegetables. "
        "It can be served as nigiri, maki rolls, or other styles. "
        "It is known for its fresh ingredients and precise preparation.",
    },
    "tacos": {
        "calories_per_100g": 226,
        "ingredients": [
            "corn tortilla",
            "meat",
            "onion",
            "cilantro",
            "salsa",
            "lime",
            "cheese",
        ],
        "description": "Tacos are Mexican dishes consisting of a tortilla folded around a filling. "
        "Fillings can include beef, pork, chicken, fish, or vegetables, topped with onion, cilantro, and salsa. "
        "They are eaten by hand and are a staple of Mexican street food.",
    },
    "takoyaki": {
        "calories_per_100g": 180,
        "ingredients": [
            "flour",
            "octopus",
            "egg",
            "dashi",
            "scallions",
            "pickled ginger",
            "bonito flakes",
            "mayonnaise",
        ],
        "description": "Takoyaki are Japanese ball-shaped snacks made from a batter containing diced octopus. "
        "They are cooked in a special moulded pan and topped with takoyaki sauce, mayonnaise, and bonito flakes. "
        "They are a popular street food, especially in Osaka.",
    },
    "tiramisu": {
        "calories_per_100g": 283,
        "ingredients": [
            "ladyfingers",
            "mascarpone",
            "espresso",
            "cocoa powder",
            "eggs",
            "sugar",
        ],
        "description": "Tiramisu is an Italian dessert made with coffee-soaked ladyfingers and mascarpone cream. "
        "It is layered and dusted with cocoa powder, then chilled until set. "
        "The name means 'pick me up', referring to the coffee and cocoa flavours.",
    },
    "tuna_tartare": {
        "calories_per_100g": 130,
        "ingredients": [
            "raw tuna",
            "soy sauce",
            "sesame oil",
            "scallions",
            "ginger",
            "avocado",
            "sesame seeds",
        ],
        "description": "Tuna tartare is a dish of finely chopped raw tuna mixed with seasonings. "
        "It often includes soy sauce, sesame oil, ginger, and scallions, and may be served with avocado. "
        "It is a light, elegant appetiser common in modern seafood restaurants.",
    },
    "waffles": {
        "calories_per_100g": 291,
        "ingredients": [
            "flour",
            "milk",
            "eggs",
            "butter",
            "sugar",
            "baking powder",
            "maple syrup",
        ],
        "description": "Waffles are batter-based cakes cooked in a waffle iron, giving them a distinctive grid pattern. "
        "They are crisp on the outside and soft inside, and are often served with syrup, butter, or fruit. "
        "They are a popular breakfast food in many countries.",
    },
}

# Safely look up a food category. Returns None if the category is missing.
def get_food_info(category: str) -> FoodInfoEntry | None:
    """Return nutrition information for a category, or ``None`` if unknown."""
    return FOOD_INFO.get(category)


__all__ = ["FOOD_INFO", "FoodInfoEntry", "get_food_info"] # Specify which objects can be imported from this file.