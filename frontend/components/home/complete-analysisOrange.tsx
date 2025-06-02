"use client"

import { Form, FormControl, FormField, FormItem, FormLabel, FormMessage } from "@/components/ui/form"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { useForm } from "react-hook-form"
import { zodResolver } from "@hookform/resolvers/zod"
import { z } from "zod"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Loader2, CheckCircle, AlertCircle, Calculator, TrendingUp, User, BarChart3 } from "lucide-react"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { Slider } from "@/components/ui/slider"
import {
  AlertDialog,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog"

const formSchema = z.object({
  age: z.number().min(17).max(90),
  capital_gain: z.number().min(0).max(99999),
  capital_loss: z.number().min(0).max(4356),
  hours_per_week: z.number().min(1).max(99),

  workclass: z
    .enum([
      "Federal-gov",
      "Local-gov", 
      "Never-worked",
      "Private",
      "Self-emp-inc",
      "Self-emp-not-inc",
      "State-gov",
      "Without-pay"
    ]),

  education: z
    .enum([
      "Preschool",
      "1st-4th",
      "5th-6th", 
      "7th-8th",
      "9th",
      "10th",
      "11th",
      "12th",
      "HS-grad",
      "Some-college",
      "Assoc-voc",
      "Assoc-acdm",
      "Bachelors",
      "Masters",
      "Prof-school",
      "Doctorate"
    ]),

  marital_status: z
    .enum([
      "Divorced",
      "Married-AF-spouse",
      "Married-civ-spouse",
      "Married-spouse-absent",
      "Never-married",
      "Separated",
      "Widowed"
    ]),

  occupation: z
    .enum([
      "Adm-clerical",
      "Armed-Forces",
      "Craft-repair",
      "Exec-managerial",
      "Farming-fishing",
      "Handlers-cleaners",
      "Machine-op-inspct",
      "Other-service",
      "Priv-house-serv",
      "Prof-specialty",
      "Protective-serv",
      "Sales",
      "Tech-support",
      "Transport-moving"
    ]),

  relationship: z
    .enum([
      "Husband",
      "Not-in-family",
      "Other-relative",
      "Own-child",
      "Unmarried",
      "Wife"
    ]),

  race: z
    .enum([
      "Amer-Indian-Eskimo",
      "Asian-Pac-Islander", 
      "Black",
      "Other",
      "White"
    ]),

  sex: z.enum(["Female", "Male"]),

  native_country: z
    .enum([
      "Cambodia", "Canada", "China", "Columbia", "Cuba", "Dominican-Republic",
      "Ecuador", "El-Salvador", "England", "France", "Germany", "Greece",
      "Guatemala", "Haiti", "Holand-Netherlands", "Honduras", "Hong", "Hungary",
      "India", "Iran", "Ireland", "Italy", "Jamaica", "Japan", "Laos", "Mexico",
      "Nicaragua", "Outlying-US(Guam-USVI-etc)", "Peru", "Philippines", "Poland",
      "Portugal", "Puerto-Rico", "Scotland", "South", "Taiwan", "Thailand",
      "Trinadad&Tobago", "United-States", "Vietnam", "Yugoslavia"
    ]),
})

type FormSchema = z.infer<typeof formSchema>
type FormKeys = keyof FormSchema

type PredictionResult = {
  prediction: string;
  probabilities: { "0": number; "1": number };
  probability: number;
}

export function CompleteAnalysisOrange() {
  const [open, setOpen] = useState(false)
  const [predictionResult, setPredictionResult] = useState<PredictionResult | null>(null)
  const [loading, setLoading] = useState(false)

  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      age: 0,
      capital_gain: 0,
      capital_loss: 0,
      hours_per_week: 0,
      workclass: undefined,
      education: undefined,
      marital_status: undefined,
      occupation: undefined,
      relationship: undefined,
      race: undefined,
      sex: undefined,
      native_country: undefined,
    },
  })

  async function onSubmit(values: z.infer<typeof formSchema>) {
    setLoading(true)
    try {
      console.log("values", values)
      const response = await fetch("http://localhost:5000/api/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(values),
      })
      const data = await response.json()
      console.log(data)
      setPredictionResult(data)
      setOpen(true)
    } catch (error) {
      console.error("Error:", error)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-orange-50 to-amber-50 p-4">
      <div className="max-w-6xl mx-auto">
        <Card className="border-orange-200 shadow-lg">
          <CardHeader className="bg-gradient-to-r from-orange-100 to-amber-100 border-b border-orange-200">
            <div className="flex items-center space-x-3">
              <Calculator className="h-8 w-8 text-orange-600" />
              <div>
                <CardTitle className="text-2xl text-orange-900">Complete Prediction Analysis</CardTitle>
                <CardDescription className="text-orange-700">
                  Fill in demographic information to predict income level
                </CardDescription>
              </div>
            </div>
          </CardHeader>
          <CardContent className="p-6">
            <Form {...form}>
              <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-8">
                {/* Numerical features section */}
                <div className="space-y-6">
                  <div className="flex items-center space-x-3 pb-4 border-b border-orange-200">
                    <TrendingUp className="h-6 w-6 text-orange-600" />
                    <h3 className="text-2xl font-bold text-orange-900">Numerical Features</h3>
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <FormField
                      control={form.control}
                      name="age"
                      render={({ field }) => (
                        <FormItem className="bg-orange-50 p-4 rounded-lg border border-orange-200">
                          <FormLabel className="text-orange-900 font-semibold">Age</FormLabel>
                          <FormControl>
                            <div className="space-y-3">
                              <Slider
                                min={17}
                                max={90}
                                step={1}
                                value={[field.value]}
                                onValueChange={(value) => field.onChange(value[0])}
                                className="[&_[role=slider]]:bg-orange-500 [&_[role=slider]]:border-orange-600"
                              />
                              <Input
                                type="number"
                                value={field.value}
                                onChange={(e) => field.onChange(Number(e.target.value))}
                                className="border-orange-300 focus:border-orange-500 focus:ring-orange-500"
                              />
                              <p className="text-sm text-orange-600">Min: 17 | Max: 90</p>
                            </div>
                          </FormControl>
                          <FormMessage />
                        </FormItem>
                      )}
                    />

                    <FormField
                      control={form.control}
                      name="capital_gain"
                      render={({ field }) => (
                        <FormItem className="bg-orange-50 p-4 rounded-lg border border-orange-200">
                          <FormLabel className="text-orange-900 font-semibold">Capital Gains</FormLabel>
                          <FormControl>
                            <div className="space-y-3">
                              <Slider
                                min={0}
                                max={99999}
                                step={1}
                                value={[field.value]}
                                onValueChange={(value) => field.onChange(value[0])}
                                className="[&_[role=slider]]:bg-orange-500 [&_[role=slider]]:border-orange-600"
                              />
                              <Input
                                type="number"
                                value={field.value}
                                onChange={(e) => field.onChange(Number(e.target.value))}
                                className="border-orange-300 focus:border-orange-500 focus:ring-orange-500"
                              />
                              <p className="text-sm text-orange-600">Min: 0 | Max: 99999</p>
                            </div>
                          </FormControl>
                          <FormMessage />
                        </FormItem>
                      )}
                    />

                    <FormField
                      control={form.control}
                      name="capital_loss"
                      render={({ field }) => (
                        <FormItem className="bg-orange-50 p-4 rounded-lg border border-orange-200">
                          <FormLabel className="text-orange-900 font-semibold">Capital Loss</FormLabel>
                          <FormControl>
                            <div className="space-y-3">
                              <Slider
                                min={0}
                                max={4356}
                                step={1}
                                value={[field.value]}
                                onValueChange={(value) => field.onChange(value[0])}
                                className="[&_[role=slider]]:bg-orange-500 [&_[role=slider]]:border-orange-600"
                              />
                              <Input
                                type="number"
                                value={field.value}
                                onChange={(e) => field.onChange(Number(e.target.value))}
                                className="border-orange-300 focus:border-orange-500 focus:ring-orange-500"
                              />
                              <p className="text-sm text-orange-600">Min: 0 | Max: 4356</p>
                            </div>
                          </FormControl>
                          <FormMessage />
                        </FormItem>
                      )}
                    />

                    <FormField
                      control={form.control}
                      name="hours_per_week"
                      render={({ field }) => (
                        <FormItem className="bg-orange-50 p-4 rounded-lg border border-orange-200">
                          <FormLabel className="text-orange-900 font-semibold">Hours per Week</FormLabel>
                          <FormControl>
                            <div className="space-y-3">
                              <Slider
                                min={1}
                                max={99}
                                step={1}
                                value={[field.value]}
                                onValueChange={(value) => field.onChange(value[0])}
                                className="[&_[role=slider]]:bg-orange-500 [&_[role=slider]]:border-orange-600"
                              />
                              <Input
                                type="number"
                                value={field.value}
                                onChange={(e) => field.onChange(Number(e.target.value))}
                                className="border-orange-300 focus:border-orange-500 focus:ring-orange-500"
                              />
                              <p className="text-sm text-orange-600">Min: 1 | Max: 99</p>
                            </div>
                          </FormControl>
                          <FormMessage />
                        </FormItem>
                      )}
                    />
                  </div>
                </div>

                {/* Categorical features section */}
                <div className="space-y-6">
                  <div className="flex items-center space-x-3 pb-4 border-b border-orange-200">
                    <User className="h-6 w-6 text-orange-600" />
                    <h3 className="text-2xl font-bold text-orange-900">Categorical Features</h3>
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                    
                    <FormField
                      control={form.control}
                      name="workclass"
                      render={({ field }) => (
                        <FormItem className="bg-orange-50 p-4 rounded-lg border border-orange-200">
                          <FormLabel className="text-orange-900 font-semibold">Work Class</FormLabel>
                          <FormControl>
                            <RadioGroup
                              onValueChange={field.onChange}
                              defaultValue={field.value}
                              className="flex flex-col space-y-2"
                            >
                              {[
                                { value: "Federal-gov", label: "Federal Government" },
                                { value: "Local-gov", label: "Local Government" },
                                { value: "Never-worked", label: "Never Worked" },
                                { value: "Private", label: "Private Sector" },
                                { value: "Self-emp-inc", label: "Self Employed (Incorporated)" },
                                { value: "Self-emp-not-inc", label: "Self Employed (Not Incorporated)" },
                                { value: "State-gov", label: "State Government" },
                                { value: "Without-pay", label: "Without Pay" },
                              ].map((option) => (
                                <FormItem key={option.value} className="flex items-center space-x-3 space-y-0">
                                  <FormControl>
                                    <RadioGroupItem
                                      value={option.value}
                                      className="border-orange-400 text-orange-600 focus:ring-orange-500"
                                    />
                                  </FormControl>
                                  <FormLabel className="font-normal text-orange-800 text-sm">{option.label}</FormLabel>
                                </FormItem>
                              ))}
                            </RadioGroup>
                          </FormControl>
                          <FormMessage />
                        </FormItem>
                      )}
                    />

                    <FormField
                      control={form.control}
                      name="education"
                      render={({ field }) => (
                        <FormItem className="bg-orange-50 p-4 rounded-lg border border-orange-200">
                          <FormLabel className="text-orange-900 font-semibold">Education</FormLabel>
                          <FormControl>
                            <RadioGroup
                              onValueChange={field.onChange}
                              defaultValue={field.value}
                              className="flex flex-col space-y-2 max-h-64 overflow-y-auto"
                            >
                              {[
                                { value: "Preschool", label: "Preschool" },
                                { value: "1st-4th", label: "1st-4th Grade" },
                                { value: "5th-6th", label: "5th-6th Grade" },
                                { value: "7th-8th", label: "7th-8th Grade" },
                                { value: "9th", label: "9th Grade" },
                                { value: "10th", label: "10th Grade" },
                                { value: "11th", label: "11th Grade" },
                                { value: "12th", label: "12th Grade" },
                                { value: "HS-grad", label: "High School Graduate" },
                                { value: "Some-college", label: "Some College" },
                                { value: "Assoc-voc", label: "Associate Degree (Vocational)" },
                                { value: "Assoc-acdm", label: "Associate Degree (Academic)" },
                                { value: "Bachelors", label: "Bachelor's Degree" },
                                { value: "Masters", label: "Master's Degree" },
                                { value: "Prof-school", label: "Professional School" },
                                { value: "Doctorate", label: "Doctorate" },
                              ].map((option) => (
                                <FormItem key={option.value} className="flex items-center space-x-3 space-y-0">
                                  <FormControl>
                                    <RadioGroupItem
                                      value={option.value}
                                      className="border-orange-400 text-orange-600 focus:ring-orange-500"
                                    />
                                  </FormControl>
                                  <FormLabel className="font-normal text-orange-800 text-sm">{option.label}</FormLabel>
                                </FormItem>
                              ))}
                            </RadioGroup>
                          </FormControl>
                          <FormMessage />
                        </FormItem>
                      )}
                    />

                    <FormField
                      control={form.control}
                      name="marital_status"
                      render={({ field }) => (
                        <FormItem className="bg-orange-50 p-4 rounded-lg border border-orange-200">
                          <FormLabel className="text-orange-900 font-semibold">Marital Status</FormLabel>
                          <FormControl>
                            <RadioGroup
                              onValueChange={field.onChange}
                              defaultValue={field.value}
                              className="flex flex-col space-y-2"
                            >
                              {[
                                { value: "Divorced", label: "Divorced" },
                                { value: "Married-AF-spouse", label: "Married (Armed Forces Spouse)" },
                                { value: "Married-civ-spouse", label: "Married (Civilian Spouse)" },
                                { value: "Married-spouse-absent", label: "Married (Spouse Absent)" },
                                { value: "Never-married", label: "Never Married" },
                                { value: "Separated", label: "Separated" },
                                { value: "Widowed", label: "Widowed" },
                              ].map((option) => (
                                <FormItem key={option.value} className="flex items-center space-x-3 space-y-0">
                                  <FormControl>
                                    <RadioGroupItem
                                      value={option.value}
                                      className="border-orange-400 text-orange-600 focus:ring-orange-500"
                                    />
                                  </FormControl>
                                  <FormLabel className="font-normal text-orange-800 text-sm">{option.label}</FormLabel>
                                </FormItem>
                              ))}
                            </RadioGroup>
                          </FormControl>
                          <FormMessage />
                        </FormItem>
                      )}
                    />

                    <FormField
                      control={form.control}
                      name="occupation"
                      render={({ field }) => (
                        <FormItem className="bg-orange-50 p-4 rounded-lg border border-orange-200 md:col-span-2 lg:col-span-1">
                          <FormLabel className="text-orange-900 font-semibold">Occupation</FormLabel>
                          <FormControl>
                            <RadioGroup
                              onValueChange={field.onChange}
                              defaultValue={field.value}
                              className="grid grid-cols-1 gap-2 max-h-64 overflow-y-auto"
                            >
                              {[
                                { value: "Adm-clerical", label: "Administrative/Clerical" },
                                { value: "Armed-Forces", label: "Armed Forces" },
                                { value: "Craft-repair", label: "Craft/Repair" },
                                { value: "Exec-managerial", label: "Executive/Managerial" },
                                { value: "Farming-fishing", label: "Farming/Fishing" },
                                { value: "Handlers-cleaners", label: "Handlers/Cleaners" },
                                { value: "Machine-op-inspct", label: "Machine Operator/Inspector" },
                                { value: "Other-service", label: "Other Service" },
                                { value: "Priv-house-serv", label: "Private Household Service" },
                                { value: "Prof-specialty", label: "Professional Specialty" },
                                { value: "Protective-serv", label: "Protective Service" },
                                { value: "Sales", label: "Sales" },
                                { value: "Tech-support", label: "Technical Support" },
                                { value: "Transport-moving", label: "Transportation/Moving" },
                              ].map((option) => (
                                <FormItem key={option.value} className="flex items-center space-x-3 space-y-0">
                                  <FormControl>
                                    <RadioGroupItem
                                      value={option.value}
                                      className="border-orange-400 text-orange-600 focus:ring-orange-500"
                                    />
                                  </FormControl>
                                  <FormLabel className="font-normal text-orange-800 text-sm">{option.label}</FormLabel>
                                </FormItem>
                              ))}
                            </RadioGroup>
                          </FormControl>
                          <FormMessage />
                        </FormItem>
                      )}
                    />

                    <FormField
                      control={form.control}
                      name="relationship"
                      render={({ field }) => (
                        <FormItem className="bg-orange-50 p-4 rounded-lg border border-orange-200">
                          <FormLabel className="text-orange-900 font-semibold">Relationship</FormLabel>
                          <FormControl>
                            <RadioGroup
                              onValueChange={field.onChange}
                              defaultValue={field.value}
                              className="flex flex-col space-y-2"
                            >
                              {[
                                { value: "Husband", label: "Husband" },
                                { value: "Not-in-family", label: "Not in Family" },
                                { value: "Other-relative", label: "Other Relative" },
                                { value: "Own-child", label: "Own Child" },
                                { value: "Unmarried", label: "Unmarried" },
                                { value: "Wife", label: "Wife" },
                              ].map((option) => (
                                <FormItem key={option.value} className="flex items-center space-x-3 space-y-0">
                                  <FormControl>
                                    <RadioGroupItem
                                      value={option.value}
                                      className="border-orange-400 text-orange-600 focus:ring-orange-500"
                                    />
                                  </FormControl>
                                  <FormLabel className="font-normal text-orange-800">{option.label}</FormLabel>
                                </FormItem>
                              ))}
                            </RadioGroup>
                          </FormControl>
                          <FormMessage />
                        </FormItem>
                      )}
                    />

                    <FormField
                      control={form.control}
                      name="race"
                      render={({ field }) => (
                        <FormItem className="bg-orange-50 p-4 rounded-lg border border-orange-200">
                          <FormLabel className="text-orange-900 font-semibold">Race</FormLabel>
                          <FormControl>
                            <RadioGroup
                              onValueChange={field.onChange}
                              defaultValue={field.value}
                              className="flex flex-col space-y-2"
                            >
                              {[
                                { value: "Amer-Indian-Eskimo", label: "American Indian/Eskimo" },
                                { value: "Asian-Pac-Islander", label: "Asian/Pacific Islander" },
                                { value: "Black", label: "Black" },
                                { value: "Other", label: "Other" },
                                { value: "White", label: "White" },
                              ].map((option) => (
                                <FormItem key={option.value} className="flex items-center space-x-3 space-y-0">
                                  <FormControl>
                                    <RadioGroupItem
                                      value={option.value}
                                      className="border-orange-400 text-orange-600 focus:ring-orange-500"
                                    />
                                  </FormControl>
                                  <FormLabel className="font-normal text-orange-800">{option.label}</FormLabel>
                                </FormItem>
                              ))}
                            </RadioGroup>
                          </FormControl>
                          <FormMessage />
                        </FormItem>
                      )}
                    />

                    <FormField
                      control={form.control}
                      name="sex"
                      render={({ field }) => (
                        <FormItem className="bg-orange-50 p-4 rounded-lg border border-orange-200">
                          <FormLabel className="text-orange-900 font-semibold">Sex</FormLabel>
                          <FormControl>
                            <RadioGroup
                              onValueChange={field.onChange}
                              defaultValue={field.value}
                              className="flex flex-col space-y-2"
                            >
                              {[
                                { value: "Female", label: "Female" },
                                { value: "Male", label: "Male" },
                              ].map((option) => (
                                <FormItem key={option.value} className="flex items-center space-x-3 space-y-0">
                                  <FormControl>
                                    <RadioGroupItem
                                      value={option.value}
                                      className="border-orange-400 text-orange-600 focus:ring-orange-500"
                                    />
                                  </FormControl>
                                  <FormLabel className="font-normal text-orange-800">{option.label}</FormLabel>
                                </FormItem>
                              ))}
                            </RadioGroup>
                          </FormControl>
                          <FormMessage />
                        </FormItem>
                      )}
                    />

                    <FormField
                      control={form.control}
                      name="native_country"
                      render={({ field }) => (
                        <FormItem className="bg-orange-50 p-4 rounded-lg border border-orange-200 md:col-span-2 lg:col-span-1">
                          <FormLabel className="text-orange-900 font-semibold">Native Country</FormLabel>
                          <FormControl>
                            <RadioGroup
                              onValueChange={field.onChange}
                              defaultValue={field.value}
                              className="grid grid-cols-2 gap-2 max-h-64 overflow-y-auto"
                            >
                              {[
                                "Cambodia", "Canada", "China", "Columbia", "Cuba", "Dominican-Republic",
                                "Ecuador", "El-Salvador", "England", "France", "Germany", "Greece",
                                "Guatemala", "Haiti", "Holand-Netherlands", "Honduras", "Hong", "Hungary",
                                "India", "Iran", "Ireland", "Italy", "Jamaica", "Japan", "Laos", "Mexico",
                                "Nicaragua", "Outlying-US(Guam-USVI-etc)", "Peru", "Philippines", "Poland",
                                "Portugal", "Puerto-Rico", "Scotland", "South", "Taiwan", "Thailand",
                                "Trinadad&Tobago", "United-States", "Vietnam", "Yugoslavia"
                              ].map((country) => (
                                <FormItem key={country} className="flex items-center space-x-3 space-y-0">
                                  <FormControl>
                                    <RadioGroupItem
                                      value={country}
                                      className="border-orange-400 text-orange-600 focus:ring-orange-500"
                                    />
                                  </FormControl>
                                  <FormLabel className="font-normal text-orange-800 text-xs">{country}</FormLabel>
                                </FormItem>
                              ))}
                            </RadioGroup>
                          </FormControl>
                          <FormMessage />
                        </FormItem>
                      )}
                    />
                  </div>
                </div>

                <div className="flex justify-center pt-6">
                  <Button
                    type="submit"
                    disabled={loading}
                    className="bg-gradient-to-r from-orange-500 to-amber-500 hover:from-orange-600 hover:to-amber-600 text-white font-semibold px-8 py-3 text-lg shadow-lg"
                    onClick={() => {
                      const formValues = form.getValues()
                      const missingValues = Object.entries(formValues)
                        .filter(([_, value]) => value === undefined || value === 0)
                        .map(([key]) => key)
                      console.log("Missing values:", missingValues)
                    }}
                  >
                    {loading && <Loader2 className="mr-2 h-5 w-5 animate-spin" />}
                    {loading ? "Analysis in progress..." : "Run Prediction"}
                  </Button>
                </div>
              </form>
            </Form>

            <AlertDialog open={open} onOpenChange={setOpen}>
              <AlertDialogContent className="border-orange-200">
                <AlertDialogHeader>
                  <AlertDialogTitle className="text-orange-900 flex items-center space-x-2">
                    {predictionResult && predictionResult.prediction === ">50K" ? (
                      <CheckCircle className="h-6 w-6 text-green-500" />
                    ) : (
                      <AlertCircle className="h-6 w-6 text-orange-500" />
                    )}
                    <span>Prediction Result</span>
                  </AlertDialogTitle>
                  <AlertDialogDescription asChild>
                    <div className="text-lg">
                      {predictionResult ? (
                        <div className="space-y-3">
                          <div className="text-center">
                            <div
                              className={`text-2xl font-bold p-4 rounded-lg ${
                                predictionResult.prediction === ">50K"
                                  ? "bg-green-100 text-green-800 border border-green-200"
                                  : "bg-orange-100 text-orange-800 border border-orange-200"
                              }`}
                            >
                              Predicted income: {predictionResult.prediction}
                            </div>
                          </div>
                          <p className="text-orange-700">
                            The model predicts that this person will have an income{" "}
                            <strong>
                              {predictionResult.prediction === ">50K" ? "greater than $50,000" : "less than or equal to $50,000"}
                            </strong>{" "}
                            per year.
                          </p>
                          
                          {/* Probabilités justificatives */}
                          <div className="mt-4 p-4 bg-gray-50 rounded-lg border border-gray-200">
                            <h4 className="font-semibold text-gray-800 mb-3 flex items-center">
                              <BarChart3 className="h-4 w-4 mr-2" />
                              Detailed probabilities
                            </h4>
                            <div className="space-y-3">
                              <div className="flex justify-between items-center">
                                <span className="text-sm text-gray-600">Income ≤ $50K :</span>
                                <div className="flex items-center space-x-2">
                                  <div className="w-24 bg-gray-200 rounded-full h-2">
                                    <div 
                                      className="bg-orange-500 h-2 rounded-full" 
                                      style={{ width: `${(predictionResult.probabilities["0"] * 100)}%` }}
                                    ></div>
                                  </div>
                                  <span className="text-sm font-medium text-gray-800">
                                    {(predictionResult.probabilities["0"] * 100).toFixed(1)}%
                                  </span>
                                </div>
                              </div>
                              <div className="flex justify-between items-center">
                                <span className="text-sm text-gray-600">Income &gt; $50K :</span>
                                <div className="flex items-center space-x-2">
                                  <div className="w-24 bg-gray-200 rounded-full h-2">
                                    <div 
                                      className="bg-green-500 h-2 rounded-full" 
                                      style={{ width: `${(predictionResult.probabilities["1"] * 100)}%` }}
                                    ></div>
                                  </div>
                                  <span className="text-sm font-medium text-gray-800">
                                    {(predictionResult.probabilities["1"] * 100).toFixed(1)}%
                                  </span>
                                </div>
                              </div>
                            </div>
                            <div className="mt-3 pt-3 border-t border-gray-200">
                              <p className="text-xs text-gray-500">
                                Prediction confidence: <strong>{(predictionResult.probability * 100).toFixed(1)}%</strong>
                              </p>
                            </div>
                          </div>
                        </div>
                      ) : (
                        <div className="text-center text-orange-600">No prediction available.</div>
                      )}
                    </div>
                  </AlertDialogDescription>
                </AlertDialogHeader>
                <AlertDialogFooter>
                  <div className="flex justify-between w-full">
                    <Button
                      variant="outline"
                      onClick={() => window.location.href = '/feature-importance'}
                      className="border-orange-300 text-orange-700 hover:bg-orange-50"
                    >
                      <BarChart3 className="mr-2 h-4 w-4" />
                      See feature importance
                    </Button>
                    <AlertDialogCancel
                      onClick={() => setOpen(false)}
                      className="border-orange-300 text-orange-700 hover:bg-orange-50"
                    >
                      Close
                    </AlertDialogCancel>
                  </div>
                </AlertDialogFooter>
              </AlertDialogContent>
            </AlertDialog>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
