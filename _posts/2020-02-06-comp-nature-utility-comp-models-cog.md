---
title: About the nature and utility of computational models of cognition
published: true
---

<iframe src="https://github.com/sponsors/pabloinsente/card" title="Sponsor pabloinsente" height="112" width="300" style="border: 0;"></iframe>

## What are computational models of cognition? 

To explain what computational models of cognition are, It is useful to state first what they are not: *computational models of cognition are not exact replicas of the human mind*, in the same manner, that city-maps are not exact replicas of real cities. The only way to create an exact map of a city is by rebuilding the city all over again until you get an inch by inch match between the city and your map, point at which the map becomes useless, as famously exposed by Jorge Luis Borges on "Del Rigor de la Ciencia" ("On Exactitude of Science", 1998):



> …In that Empire, the Art of Cartography attained such Perfection that the map of a single Province occupied the entirety of a City, and the map of the Empire, the entirety of a Province. In time, those Unconscionable Maps no longer satisfied, and the Cartographers Guilds struck a Map of the Empire whose size was that of the Empire, and which coincided point for point with it. The following Generations, who were not so fond of the Study of Cartography as their Forebears had been, saw that that vast Map was Useless, and not without some Pitilessness was it, that they delivered it up to the Inclemencies of Sun and Winters. In the Deserts of the West, still today, there are Tattered Ruins of that Map, inhabited by Animals and Beggars; in all the Land there is no other Relic of the Disciplines of Geography. 
>
> —Suarez Miranda,Viajes devarones prudentes, Libro IV,Cap. XLV, Lerida, 1658



Therefore, computational models of cognition are, in a sense, maps of cognition, or more academically, *simplified abstract representation of the mind*. Now, maps can help to understand multiple aspects of a city: the weather, political boundaries, economic regions, roads, topography, etc. Again, the same can be said about models of cognition. Some models help to understand memory, others language production, others visual perception, and so on. True, a model of language production does not need to be computational, it can be stated verbally, or graphically in a diagram. Informally, *to compute* means to take a set of inputs, manipulate those inputs using a set of operations following a sequence of instructions (or algorithm), to then produce an output. From here, we derive that computational models of cognition can be defined as *simplified abstract representations of the mind, that describe how some aspects of the mind process information in an algorithmic fashion, to produce some output* (e.g., language, inference, perception, etc). This is both more convoluted and more humble than "a replica of the mind".



## What are computational models of cognition good for?

Examining the history of computational models in cognitive science says a lot about the importance of this approach to the study of the mind and behavior. Still, is important to clarify how computational modeling is different from other approaches, and what advantages and disadvantages entail. 



Any attempt to classify the multiple approaches to the study of cognition and behavior will inevitably misrepresent their "true identity", exaggerating some aspects and neglecting others. Nonetheless, this exercise will help us to illustrate what is unique to computational approaches. Following Cronbach's perspective (1957), the two main traditions in scientific psychology are the *experimental* and the *correlational*. By experimental, he referred to laboratory-based quantitative studies. Today, we may want to add to that tradition any controlled experimental study aimed to establish causation, regardless of the setting. The correlational tradition is a bit fuzzier. About this Cronbach said: "*The correlational method, for its part, can study what man has not learned to control or can never hope to control*" (p. 672), which is a different way to say non-experimental and non-controlled studies, in any setting, using qualitative and quantitative methods. Very broad.



To begin, it is important to remember that cognition is not something you can directly observe. If we could, the chances are that many of today's challenges and controversies would be solved and that I wouldn't be writing this document. Any research study about cognition, either correlational or experimental, will proceed by measuring behavior, things like reaction time or eye-movements, and then making inferences about cognitive processes. Even if you take the perspective that cognitive processes are *literally* patterns of neural activity in the brain, studying such patterns is very limited in scope, very complicated, and very expensive. Let's say that you want to study language acquisition in early childhood. How would you approach the study of the cognitive processes operating when a child learns language? Techniques like functional magnetic resonance imaging (fMRI) are really hard to perform on children, since requires staying still in a giant noisy tube for an extended period of time. Even if you manage to introduce a child in an fMRI machine and stay still for 20 minutes (as many very clever researchers do), now you have to figure out a task that the child can perform, with some sort of remote control. True, there are some task that can be used with young children, but the question now is *what* and *how much* can you learn from having a child performing one task for 20 min in such setting. You have also to consider the fact that language acquisition is a developmental process, so multiple sessions over an extended period of time are required to gain better insight. Other techniques like electroencephalogram are less invasive and easier to perform on children, but many of the fMRI limitations persist. Does this mean that studies based on measuring behavior and brain activity are useless? Absolutely not. They both have their strengths and weaknesses. Our point was to illustrate that they have many limitations that make desirable to have an additional method to study human cognition.



Computational modeling has a lot to offer to enhance our understanding of human cognition, as a complement, not a supplement, of behavioral and brain-based studies. I want to highlight the following aspects: *isolation*, *simulation*, *simplification*, *quantification*, *practicality*, and *theory exploration*.

- **Isolation**:  in studies with human subjects, it is really hard to isolate the effect of a specific cognitive process. In correlational studies, this is simply impossible, as anyone trying to account for "confounding" factors have experienced. In experimental studies, you can isolate the "treatment" theoretically impacting a cognitive process, but the process itself. Computational models do allow for such isolation of processes, as the processes are built into the model by the modeler, and can be altered at will without touching other aspects of the model.  
- **Simulation**: computational models allow answering "what if" questions. What if I accelerate the speed at which the model learns in this particular task? Easy, just increase the "learning rate" parameter. What if I introduce "noise" in the communication among neurons in my neural network? Easy, too, just add some random noise to the weights matrix. Artificially manipulating cognitive processes in humans are possible too, for instance, by using transcranial magnetic stimulation or TMS. Unfortunately, you have little to no control over which processes are affected, unless you use very invasive methods, which usually require brain surgery. Additionally, the results of a simulation can be compared to human data, to further assess the validity of the model.
- **Simplification**: each element in a computational model needs to be explicitly defined. This fact forces the modeler to carefully select a subset of cognitive mechanisms to be incorporated into the model. As a consequence, the simplification of highly complex and interactive mental processes is achieved. Simplifying a complex system may help to better understand the role of each component. 
- **Quantification**: by its very nature, computational models allow for fine-grained quantification of the role of each mechanism in a model. This is not unique to computational models, but still is one of its main advantages. 
- **Practically**: in most instances, computational models require only two things: a computer and a modeler. More complex scenarios may require a cluster of computers and multiple modelers, but still, that is considerably simpler and cheaper, than most experimental psychology protocols, and than most nonscientific protocols. Although having human-generated data is desirable in many instances, this is not strictly required, and even if you use human data, it is common practice to use secondary sources of data (data collected in previous studies, sometimes by other research groups). In many cases, computational models can be tested with synthetic data, or with no data at all.
- **Theory exploration**: truth to be told, any research approach would allow for theory exploration. However, the computational modeling approach has the advantage of access to secondary data, and synthetic data, lowering the bar for examining the implications of hypothesized cognitive mechanism. In a way, it allows for rapid iteration without having to necessarily design a whole data collection process from scratch every time.  



## What are the limitations of computational models of cognition?

By now, you may have devised many objections and weaknesses of the computational modeling approach. I'll limit myself to mention a few that are particularly important to mention in my opinion: *oversimplification*, *overcomplexity*, *falsifiability*, and *technical complexity*:



- **Oversimplification**: this may sound contradicting, because before I highlighted *simplification* as one of the main advantages of computational models. I see simplification as a double-edged sword: on the one hand, it can help to see the bigger picture, but in the other, it can convey a distorted image of a phenomenon, to the point that hinders comprehension rather the enhancing it. 
- **Overcomplexity**: related to the previous point, I also argued that models help to handle complexity. But they also allow the modeler to make the model as complex as it wants. Very complex, with lots of free parameters, can fit or approximate almost any human-generated data, giving the impression that they are a good representation of the cognitive mechanism at play. This may or may not be the truth, but the issues are that now understanding the model becomes almost impossible.
- **Falsifiability**: Popper's falsifiability criteria (2005), refers to the idea that for a scientific hypothesis to be valid, it has to be formulated in a manner in which it can be proved wrong. In other words, you have specified in advance what piece of evidence would prove your model wrong. It turns out that proving computational models wrong is quite hard, precisely because you can make it increasingly complex. This does not mean that it is impossible. For example, Palminteri, Wyart, & Koechlin (2017) have proposed utilizing the *generative performance criteria* for model falsification, this is: "*the ability of a given model to generate the data. The generative performance is evaluated by comparing model simulations to the actual data. For this comparison both frequentist and Bayesian statistics can be used*" (p. 426).
- **Technical complexity**: there is a significant barrier to computational modeling: mathematical and programming skills. Building computational models require a firm understanding of the mathematics involved in the model definition and non-trivial programming skills. Historically, mathematical and computational skills have not been part of the core training of researchers in psychological sciences. If you examine the background of most well-known cognitive modelers, you will find that the vast majority of them had additional training in mathematics and computer science, and some were straight mathematicians or computer scientists. At present, there are reasons to be optimistic, as tools that simplify the building of computational models becomes more available, and computational skills are added to the formative curriculum of young researchers. 

## References

Borges, J. L., Hurley, A., & Hurley, A. (1998). Collected fictions. Penguin Books New York.

Cronbach, L. J. (1957). The two disciplines of scientific psychology. American Psychologist, 12(11), 671.

Palminteri, S., Wyart, V., & Koechlin, E. (2017). The Importance of Falsification in Computational Cognitive Modeling. Trends in Cognitive Sciences, 21(6), 425–433. https://doi.org/10.1016/j.tics.2017.03.011

Popper, K. (2005). The logic of scientific discovery. Routledge.
