### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ bfc35f51-da08-4ef8-8f10-280b6dc52c0f
import Pkg;

# ╔═╡ 1bfbd550-b644-4507-a2a1-841662dd7be8
begin
	Pkg.add("DecisionTree")
	Pkg.add("MLJDecisionTreeInterface")
	Pkg.add("PlutoUI")
	Pkg.add("HypertextLiteral")
	Pkg.add("DelimitedFiles")
	Pkg.add("DataFrames")
	Pkg.add("CategoricalArrays")
	Pkg.add("Statistics")
	Pkg.add("Distances")
	Pkg.add("StatsPlots")
	Pkg.add("Compose")
	Pkg.add("ColorSchemes")
	Pkg.add("Distributions")
	Pkg.add("StatsBase")
	Pkg.add("MLJBase")
end;

# ╔═╡ 9e6f1462-67b7-11ee-29df-01ca44a58aec
using Plots, PlutoUI, HypertextLiteral, DelimitedFiles, DataFrames, CategoricalArrays, Statistics, Distances, StatsPlots, Compose, ColorSchemes, Distributions

# ╔═╡ 3034c57e-709f-4f7d-9d41-5abdc9def603
using StatsBase, MLJBase, MLJDecisionTreeInterface, Random, DecisionTree, MLJ, TreeRecipe, StableRNGs, BetaML

# ╔═╡ c7fd0812-f0d1-49ad-baa2-3aa09554e230
begin 
	data1 = DataFrame(readdlm("imputed_va.csv", ',', header=false), :auto);
	data2 = DataFrame(readdlm("imputed_swiss.csv", ',', header=false), :auto);
	data3 = DataFrame(readdlm("imputed_hungarian.csv", ',', header=false), :auto);

	idata = vcat(data1[2:end,:],data2[2:end,:],data3[2:end,:])

	y2 = copy(idata[:,14])
	
		idata = coerce(idata, 
	:x1      => MLJBase.Continuous,
	:x2      => OrderedFactor,
	:x3      => OrderedFactor,
	:x4      => MLJBase.Continuous,
	:x5      => MLJBase.Continuous,
	:x6      => OrderedFactor,
	:x7      => OrderedFactor,
	:x8      => MLJBase.Continuous,
	:x9      => OrderedFactor,
	:x10     => MLJBase.Continuous,
	:x12     => OrderedFactor,
	:x13     => OrderedFactor,
	:x11     => OrderedFactor,
	:x14      => OrderedFactor
	)
	
	idata = rename(idata, :x1 => :"Age", :x2 => :"Sex", :x3 => :"CP", :x4 => :"RBP", :x5 => :"Chol.", :x6 => :"BSL", :x7 => :"RECG", :x8 => :"Max HR", :x9 => :"Ex. A", :x10 => :"OP", :x11 => :"ST", :x12 => :"F. Ves.", :x13 => :"Thal", :x14 => :"Disease");


end;

# ╔═╡ ce0e7766-82ec-4c2c-b610-c79674747bea
begin 
	X1 = idata[:,(1:13)]
	y = idata[:,14]
end;

# ╔═╡ c45ff273-e53e-4b69-9c25-6e500b7f22b4
begin
    Random.seed!(411)
    tr_inds, te_inds = MLJBase.partition(1:MLJ.nrows(X1), 0.7, shuffle=true)
end;

# ╔═╡ a8b383e9-36c7-4971-b3fb-296dfca688e1
begin
	slider_dplots = @bind D Slider(2:1:5; show_value=true);
end

# ╔═╡ b6d3ac1c-9336-4b29-bae8-8bfda08040ae
DTree = @load DecisionTreeClassifier pkg=BetaML verbosity=0

# ╔═╡ ee6974a9-3a18-40d4-ab24-615eecbbf928
begin
	Random.seed!(411)
	tree2 = DTree(max_depth = D)
	mach2 = machine(tree2, X1, y, scitype_check_level=0)
	MLJ.fit!(mach2, rows=tr_inds)
end

# ╔═╡ 810594de-fedd-4778-a7c1-cace389b5820
begin
	model = DecisionTreeEstimator(max_depth = D)
	ŷ = BetaML.fit!(model, X1, y2) |> BetaML.mode
end;

# ╔═╡ 346523ba-4527-404f-abec-6a5aa7646f4f
begin
	feature_names = ["Age", "Sex","CP", "RBP","Chol", "BSL","RECG", "Max HR","Ex. A", "OP", "ST", "F.Ves","Thal"]
	wrapped_tree = BetaML.wrap(model, (featurenames = feature_names, ))
end

# ╔═╡ 781e3b46-93d1-46be-8625-df1a9b04b3e1
function fprs_tprs(model, train, test)
    mach = machine(model, X1, y)
    MLJ.fit!(mach; rows=train)
    predictions = MLJ.predict(mach; rows=test)
    fprs, tprs, i = roc_curve(predictions, y[test])
    return fprs, tprs
end;

# ╔═╡ 140ec89f-a77f-4b67-9383-922ce028c7b1
function cross_validated_fprs_tprs(model)
	rng = StableRNG(7)
    resampling = CV(; nfolds=4, shuffle=true, rng)
    evaluations = MLJBase.evaluate(model, X1, y; measure=auc, resampling, verbosity=0)
    return map(evaluations.train_test_rows) do (train, test)
        fprs_tprs(model, train, test)
    end
end;

# ╔═╡ dac513c2-2ce5-452b-b158-737623cb707e
evaluations = MLJBase.evaluate(tree2, X1, y; measure=auc, resampling=CV(; nfolds=6, shuffle = true, rng=Random.GLOBAL_RNG), verbosity=0);

# ╔═╡ 91c11462-0ca7-4a8d-b9a9-e24e272f4d4e
aucc = round(evaluations.measurement[1], digits = 2); 

# ╔═╡ 8f4ecec6-a41a-42d1-aa14-408de600b743

begin
	tplots = Plots.plot(wrapped_tree, 1, 1.1; size = (1750,350), title = "Decision Tree Depth = $D AUC $aucc")
	Plots.annotate!(tplots, [(-7,0, Plots.text("OP -> OldPeak \nEx. A -> Exercise Induced Angina\nBSL -> Fasting Blood Sugar > 120 mg/dl \nMax HR -> Max Heart Rate\nCP -> Chest Pain Type \nF. Ves -> Number of Major Vessels colored by flourosopy \nThal -> Thal: 3 = Normal; 6 = Fixed Defect; 7 = Reversable Defect\nRBP -> Resting Blood Pressure \nChol. -> Serum Cholesterol in mg/dl", :left, 5))])
end;

# ╔═╡ 740c822c-c4a8-4a8d-8f43-0a8e062d4c40
begin 
	y_pred = MLJ.predict_mode(mach2, rows=te_inds)

	conf_matrix = MLJ.confusion_matrix(y[te_inds], y_pred)
	true_positive = conf_matrix[2, 2]
	false_negative = conf_matrix[2, 1]
	true_negative = conf_matrix[1, 1]
	false_positive = conf_matrix[1, 2]

	bar_plot = bar(["True Positive", "False Negative", "True Negative", "False Positive"],
    [true_positive, false_negative, true_negative, false_positive],
    bar_width=0.7,
	label = "",
    color=["#DF8A56", :red, "#5377C9", :red],
    xlabel="Actual Class",
    ylabel="Count",
    title="Confusion Matrix AUC $aucc", ylim = (0,250), size = (800, 450))
end; 

# ╔═╡ fdf182e1-d2ae-42de-a287-0fb6dbb1817a
dc = cross_validated_fprs_tprs(tree2);

# ╔═╡ 8a9e272e-2a5b-427b-ab68-866c5e85df8d
begin 
	fpr, tpr = dc[1]
	fpr2, tpr2 = dc[2]
	fpr3,tpr3 = dc[3]
	fpr4, tpr4 = dc[4]
end;

# ╔═╡ ea993ad0-4484-4557-9d17-f78212414b9a
begin
	rocplot = plot(fpr, tpr, label="CV ROC Curve Depth $D", xlabel="False Positive Rate", ylabel="True Positive Rate", 
	        legend=:bottomright, xlims=(0, 1), ylims=(0, 1), linewidth=2, colour=:orange, size = (800, 450))
	Plots.plot!(fpr2, tpr2, linewidth = 2, colour = :orange, label = nothing),
		Plots.plot!(fpr3, tpr3, linewidth = 2, colour = :orange, label = nothing),
		Plots.plot!(fpr4, tpr4, linewidth = 2, colour = :orange, label = nothing),
	    
	    plot!([0, 1], [0, 1], linestyle=:dash, color=:black, label="Random Guess", linewidth=2)
	    
	    title!("ROC Curve DT Depth $D || AUC = $aucc")
end;

# ╔═╡ 0d738663-daec-46a9-b30c-e6c80bcfa4b8
begin
	@info PlutoRunner.currently_running_cell_id
	PlutoUI.ExperimentalLayout.hbox([slider_dplots, tplots])
end

# ╔═╡ b316cfaa-3683-43d9-86bb-a8e1814bf732
begin
	@info PlutoRunner.currently_running_cell_id
	PlutoUI.ExperimentalLayout.hbox([rocplot, bar_plot])
end

# ╔═╡ 919a4a57-73c3-4aa6-a199-22c1a1a297b6
notebook = PlutoRunner.notebook_id[] |> string

# ╔═╡ fd707eb6-02a9-4f9a-aa91-3a025a888300
celllist=["0d738663-daec-46a9-b30c-e6c80bcfa4b8","b316cfaa-3683-43d9-86bb-a8e1814bf732"]

# ╔═╡ 0079d34a-5a8e-454a-a426-41d84378337f
dash_final_url="http://localhost:1234/edit?" * "id=$notebook&" * join(["isolated_cell_id=$cell" for cell in celllist], "&")

# ╔═╡ 6685380c-c8da-4570-af57-9a0b7e344d5b
@htl("""
<a href="$dash_final_url" style="font_size=20">Click here for the Dashboard</a>
""")

# ╔═╡ Cell order:
# ╟─6685380c-c8da-4570-af57-9a0b7e344d5b
# ╠═bfc35f51-da08-4ef8-8f10-280b6dc52c0f
# ╠═1bfbd550-b644-4507-a2a1-841662dd7be8
# ╠═9e6f1462-67b7-11ee-29df-01ca44a58aec
# ╠═3034c57e-709f-4f7d-9d41-5abdc9def603
# ╠═c7fd0812-f0d1-49ad-baa2-3aa09554e230
# ╠═ce0e7766-82ec-4c2c-b610-c79674747bea
# ╠═c45ff273-e53e-4b69-9c25-6e500b7f22b4
# ╠═a8b383e9-36c7-4971-b3fb-296dfca688e1
# ╠═b6d3ac1c-9336-4b29-bae8-8bfda08040ae
# ╠═ee6974a9-3a18-40d4-ab24-615eecbbf928
# ╠═810594de-fedd-4778-a7c1-cace389b5820
# ╠═346523ba-4527-404f-abec-6a5aa7646f4f
# ╠═8f4ecec6-a41a-42d1-aa14-408de600b743
# ╠═740c822c-c4a8-4a8d-8f43-0a8e062d4c40
# ╠═781e3b46-93d1-46be-8625-df1a9b04b3e1
# ╠═140ec89f-a77f-4b67-9383-922ce028c7b1
# ╠═dac513c2-2ce5-452b-b158-737623cb707e
# ╠═91c11462-0ca7-4a8d-b9a9-e24e272f4d4e
# ╠═fdf182e1-d2ae-42de-a287-0fb6dbb1817a
# ╠═8a9e272e-2a5b-427b-ab68-866c5e85df8d
# ╠═ea993ad0-4484-4557-9d17-f78212414b9a
# ╠═0d738663-daec-46a9-b30c-e6c80bcfa4b8
# ╠═b316cfaa-3683-43d9-86bb-a8e1814bf732
# ╠═919a4a57-73c3-4aa6-a199-22c1a1a297b6
# ╠═fd707eb6-02a9-4f9a-aa91-3a025a888300
# ╠═0079d34a-5a8e-454a-a426-41d84378337f
