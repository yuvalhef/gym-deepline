from ..core.object_analyzer import analyze_pd_dataframe
from ..core.object_to_mf_mapper import map_object_to_mf

def metafeature_generator(data, target, metaFunctions, postProcessFunctions):

    data_numpy, attributes = analyze_pd_dataframe(data, target)
    metafeatures = []
    metafeaturesNames = []


    for metaFunction in metaFunctions:

        metafeaturesComputations_Features = []
        metafeaturesComputations_Labels = []
        metafeaturesComputations_FeaturesLabels = []

        ticketsFeatures, ticketsLabels, ticketsFeaturesLabels = map_object_to_mf(attributes, metaFunction)

        if (len(ticketsFeatures) != 0):
            for ticket in ticketsFeatures:
                metafeaturesComputations_Features.append(metaFunction._calculate(data_numpy[:,ticket]))

            for postProcessFunction in postProcessFunctions:
                if (postProcessFunction.get_input_types() == metaFunction.get_output_type()):
                    if (postProcessFunction.get_input_arity() == "one") and (len(metafeaturesComputations_Features) == 1):
                        metafeatures.append(postProcessFunction._calculate(metafeaturesComputations_Features))
                        metafeaturesNames.append(
                            """Features.{mf}.{pp}""".format(
                            mf=type(metaFunction).__name__,
                            pp=type(postProcessFunction).__name__)
                        )

                    if (postProcessFunction.get_input_arity() == "two") and (len(metafeaturesComputations_Features) == 2):
                        metafeatures.append(postProcessFunction._calculate(metafeaturesComputations_Features))
                        metafeaturesNames.append(
                            """Features.{mf}.{pp}""".format(
                            mf=type(metaFunction).__name__,
                            pp=type(postProcessFunction).__name__)
                        )

                    if (postProcessFunction.get_input_arity() == "n") and (len(metafeaturesComputations_Features) >= 2):
                        metafeatures.append(postProcessFunction._calculate(metafeaturesComputations_Features))
                        metafeaturesNames.append(
                            """Features.{mf}.{pp}""".format(
                            mf=type(metaFunction).__name__,
                            pp=type(postProcessFunction).__name__)
                        )

        if (len(ticketsLabels) != 0):
            for ticket in ticketsLabels:
                metafeaturesComputations_Labels.append(metaFunction._calculate(data_numpy[:,ticket]))

            for postProcessFunction in postProcessFunctions:
                if (postProcessFunction.get_input_types() == metaFunction.get_output_type()):
                    if (postProcessFunction.get_input_arity() == "one") and (len(metafeaturesComputations_Labels) == 1):
                        metafeatures.append(postProcessFunction._calculate(metafeaturesComputations_Labels))
                        metafeaturesNames.append(
                            """Labels.{mf}.{pp}""".format(
                            mf=type(metaFunction).__name__,
                            pp=type(postProcessFunction).__name__)
                        )

                    if (postProcessFunction.get_input_arity() == "two") and (len(metafeaturesComputations_Labels) == 2):
                        metafeatures.append(postProcessFunction._calculate(metafeaturesComputations_Labels))
                        metafeaturesNames.append(
                            """Labels.{mf}.{pp}""".format(
                            mf=type(metaFunction).__name__,
                            pp=type(postProcessFunction).__name__)
                        )

                    if (postProcessFunction.get_input_arity() == "n") and (len(metafeaturesComputations_Labels) >= 2):
                        metafeatures.append(postProcessFunction._calculate(metafeaturesComputations_Labels))
                        metafeaturesNames.append(
                            """Labels.{mf}.{pp}""".format(
                            mf=type(metaFunction).__name__,
                            pp=type(postProcessFunction).__name__)
                        )

        if (len(ticketsFeaturesLabels) != 0):
            for ticket in ticketsFeaturesLabels:
                metafeaturesComputations_FeaturesLabels.append(metaFunction._calculate(data_numpy[:,ticket]))

            for postProcessFunction in postProcessFunctions:
                if (postProcessFunction.get_input_types() == metaFunction.get_output_type()):
                    if (postProcessFunction.get_input_arity() == "one") and (len(metafeaturesComputations_FeaturesLabels) == 1):
                        metafeatures.append(postProcessFunction._calculate(metafeaturesComputations_FeaturesLabels))
                        metafeaturesNames.append(
                            """FeaturesLabels.{mf}.{pp}""".format(
                            mf=type(metaFunction).__name__,
                            pp=type(postProcessFunction).__name__)
                        )

                    if (postProcessFunction.get_input_arity() == "two") and (len(metafeaturesComputations_FeaturesLabels) == 2):
                        metafeatures.append(postProcessFunction._calculate(metafeaturesComputations_FeaturesLabels))
                        metafeaturesNames.append(
                            """FeaturesLabels.{mf}.{pp}""".format(
                            mf=type(metaFunction).__name__,
                            pp=type(postProcessFunction).__name__)
                        )

                    if (postProcessFunction.get_input_arity() == "n") and (len(metafeaturesComputations_FeaturesLabels) >= 2):
                        metafeatures.append(postProcessFunction._calculate(metafeaturesComputations_FeaturesLabels))
                        metafeaturesNames.append(
                            """FeaturesLabels.{mf}.{pp}""".format(
                            mf=type(metaFunction).__name__,
                            pp=type(postProcessFunction).__name__)
                        )

    return metafeatures, metafeaturesNames
